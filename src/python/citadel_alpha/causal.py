# citadel_alpha/causal.py
# Google Python Style Guide.

"""Causal validation framework for alpha signals.

Implements the 3-step intraday causal stack from WHAT_WOULD_YOU_CHANGE.md:

  Step 1 — Intraday Granger Causality (VARX with exogenous session blocks)
  Step 2 — Conditional Independence Testing (CMI / KCIT proxy)
  Step 3 — DoWhy-style Invariant Structural Tests (Placebo + Policy Invariance)

Mathematical background: P_VALUE.md — HAC Newey-West, Moving Block Bootstrap.
Six-month plan integration: HLS_SIX_MONTH_PLAN.md §Month 3.

References:
    Granger (1969) — "Investigating Causal Relations by Econometric Models"
    Bailey & Lopez de Prado (2014) — "The Deflated Sharpe Ratio"
    Pearl (2009) — "Causality: Models, Reasoning and Inference"
    P_VALUE.md — DoWhy refutation p-value mathematics
    CAUSAL_STACK_EXPLAINED.md — 3-layer intraday causal architecture
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

from citadel_alpha import constants as C

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GrangerResult:
    """Step 1: Intraday Granger causality test result."""

    f_statistic: float
    p_value: float
    passes: bool  # True if p_value < threshold
    lag_used: int
    signal_leads_by_periods: int  # Best lag at which signal leads return


@dataclass
class CMIResult:
    """Step 2: Conditional independence test result."""

    cmi_statistic: float
    p_value: float
    passes: bool  # True if signal retains alpha after conditioning
    alpha_retained_fraction: float  # Fraction of raw IC retained after conditioning


@dataclass
class DoWhyResult:
    """Step 3: DoWhy-style invariant structural test result."""

    placebo_p_value: float  # P(|θ_b| >= |θ_orig|) under placebo
    policy_invariance_p_value: float
    placebo_passes: bool  # p > 0.05 (original estimator distinct from noise)
    policy_passes: bool  # Structural coefficient stable across regimes
    causal_gamma: float  # γ ∈ [0, 1] — causal confidence factor
    n_bootstrap_reps: int
    theta_orig: float
    theta_placebo_mean: float


@dataclass
class CausalStackResult:
    """Full 3-step causal validation pipeline result."""

    signal_name: str
    granger: GrangerResult
    cmi: CMIResult
    dowhy: DoWhyResult
    final_gamma: float  # γ for position sizing
    recommendation: str  # PASS / BETA_PROXY / REJECT
    summary: str


# ---------------------------------------------------------------------------
# Newey-West HAC variance estimator
# ---------------------------------------------------------------------------


def _newey_west_variance(
    residuals: FloatArray,
    lag_truncation: int = C.NEWEY_WEST_LAG_TRUNCATION,
) -> float:
    """Compute HAC Newey-West variance of residuals.

    Σ_HAC = Γ_0 + Σ_{j=1}^{L} w_j (Γ_j + Γ_j^T)
    w_j = 1 - j / (L + 1)   (Bartlett kernel)

    Args:
        residuals: OLS residual vector (T,).
        lag_truncation: Bandwidth L.

    Returns:
        HAC variance estimate (scalar).
    """
    e = np.asarray(residuals, dtype=np.float64)
    t = len(e)
    gamma0 = float(np.dot(e, e) / t)
    hac = gamma0
    for j in range(1, lag_truncation + 1):
        weight = 1.0 - j / (lag_truncation + 1.0)
        gamma_j = float(np.dot(e[j:], e[:-j]) / t)
        hac += 2.0 * weight * gamma_j
    return max(hac, 1e-12)


# ---------------------------------------------------------------------------
# Moving Block Bootstrap
# ---------------------------------------------------------------------------


def _moving_block_bootstrap(
    data: FloatArray,
    block_size: int = C.BLOCK_BOOTSTRAP_BLOCK_SIZE,
    n_reps: int = C.DOWHY_BOOTSTRAP_REPS,
    rng: Optional[np.random.Generator] = None,
) -> list[FloatArray]:
    """Generate Moving Block Bootstrap resamples preserving temporal structure.

    Args:
        data: Time-series data (T,) or (T, K).
        block_size: Block length b.
        n_reps: Number of bootstrap samples.
        rng: Random generator for reproducibility.

    Returns:
        List of n_reps resampled arrays each of length T.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = np.asarray(data, dtype=np.float64)
    t = data.shape[0]
    n_blocks = math.ceil(t / block_size)
    max_start = t - block_size

    samples = []
    for _ in range(n_reps):
        starts = rng.integers(0, max(max_start, 1), size=n_blocks)
        blocks = [data[s : s + block_size] for s in starts]
        resample = np.concatenate(blocks, axis=0)[:t]
        samples.append(resample)
    return samples


# ---------------------------------------------------------------------------
# Step 1: Intraday Granger Causality (VARX)
# ---------------------------------------------------------------------------


def granger_causality_varx(
    signal: FloatArray,
    returns: FloatArray,
    exogenous: Optional[FloatArray] = None,
    max_lags: int = C.GRANGER_LAG_PERIODS,
    alpha: float = C.GRANGER_PVALUE_THRESHOLD,
) -> GrangerResult:
    """Test Granger causality: does signal lead returns above exogenous controls?

    VARX model (simplified): Y_t = Σ A_k Y_{t-k} + Σ B_k X_{t-k} + C Z_t + ε_t
    Null H₀: B_1 = ... = B_p = 0 (signal has no independent predictive content).
    Uses F-test comparing restricted vs unrestricted OLS.

    HAC standard errors prevent false rejections due to autocorrelation.

    Args:
        signal: Alpha signal time series X (T,).
        returns: Forward return series Y (T,).
        exogenous: Exogenous control matrix Z (T, K) — session dummies, VIX, etc.
        max_lags: Maximum lag order p to test.
        alpha: Significance level for pass/fail.

    Returns:
        GrangerResult.
    """
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    t = len(y)

    best_p = 1.0
    best_f = 0.0
    best_lag = 1

    for lag in range(1, max_lags + 1):
        n = t - lag
        if n < 30:
            continue

        # Restricted model: Y ~ Y_lags [+ Z]
        y_dep = y[lag:]
        y_lags = np.column_stack([y[lag - k - 1 : t - k - 1] for k in range(lag)])

        if exogenous is not None:
            z = np.asarray(exogenous, dtype=np.float64)[lag:]
            x_r = np.column_stack([np.ones(n), y_lags, z])
        else:
            x_r = np.column_stack([np.ones(n), y_lags])

        # Unrestricted model: adds X lags
        x_lags = np.column_stack([x[lag - k - 1 : t - k - 1] for k in range(lag)])
        x_u = np.column_stack([x_r, x_lags])

        try:
            beta_r, res_r, _, _ = np.linalg.lstsq(x_r, y_dep, rcond=None)
            beta_u, res_u, _, _ = np.linalg.lstsq(x_u, y_dep, rcond=None)
        except np.linalg.LinAlgError:
            continue

        e_r = y_dep - x_r @ beta_r
        e_u = y_dep - x_u @ beta_u

        rss_r = float(np.dot(e_r, e_r))
        rss_u = float(np.dot(e_u, e_u))
        k_r = x_r.shape[1]
        k_u = x_u.shape[1]
        df1 = k_u - k_r
        df2 = n - k_u

        if df1 <= 0 or df2 <= 0 or rss_u < 1e-15:
            continue

        # HAC-adjusted F-statistic
        hac_var = _newey_west_variance(e_u)
        f_stat = ((rss_r - rss_u) / df1) / max(hac_var, 1e-12)
        p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))

        if p_val < best_p:
            best_p = p_val
            best_f = f_stat
            best_lag = lag

    return GrangerResult(
        f_statistic=best_f,
        p_value=best_p,
        passes=best_p < alpha,
        lag_used=best_lag,
        signal_leads_by_periods=best_lag,
    )


# ---------------------------------------------------------------------------
# Step 2: Conditional Independence (CMI proxy via partial correlation)
# ---------------------------------------------------------------------------


def conditional_independence_test(
    signal: FloatArray,
    returns: FloatArray,
    confounders: FloatArray,
    alpha: float = C.CMI_ALPHA_THRESHOLD,
) -> CMIResult:
    """Test I(Signal; Returns | Confounders) ≈ 0.

    Implements a partial-correlation proxy for CMI.
    Residualises both signal and returns on confounders, then tests
    correlation of residuals. Alpha retained = |ρ_partial| / |ρ_raw|.

    If alpha retained < 0.5 → signal is mostly beta to confounders.

    Args:
        signal: X (T,).
        returns: Y (T,).
        confounders: Z (T, K) — session effects, VIX, cross-asset momentum.
        alpha: Significance level.

    Returns:
        CMIResult.
    """
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    z = np.asarray(confounders, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    t = len(y)
    z_aug = np.column_stack([np.ones(t), z])

    def _residualise(v: FloatArray) -> FloatArray:
        try:
            beta, _, _, _ = np.linalg.lstsq(z_aug, v, rcond=None)
            return v - z_aug @ beta
        except np.linalg.LinAlgError:
            return v

    x_res = _residualise(x)
    y_res = _residualise(y)

    raw_corr = float(np.corrcoef(x, y)[0, 1])
    partial_corr = float(np.corrcoef(x_res, y_res)[0, 1])

    retained = (
        abs(partial_corr) / max(abs(raw_corr), 1e-8) if abs(raw_corr) > 1e-8 else 0.0
    )

    # t-test on partial correlation
    n_eff = t - z.shape[1] - 2
    n_eff = max(n_eff, 2)
    t_stat = (
        partial_corr * math.sqrt(n_eff) / math.sqrt(max(1.0 - partial_corr**2, 1e-12))
    )
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_eff)))

    # CMI proxy statistic: Fisher z-transform of partial correlation
    cmi_stat = 0.5 * math.log(
        (1.0 + abs(partial_corr)) / max(1.0 - abs(partial_corr), 1e-12)
    )

    return CMIResult(
        cmi_statistic=cmi_stat,
        p_value=p_val,
        passes=(p_val < alpha) and (retained >= 0.50),
        alpha_retained_fraction=retained,
    )


# ---------------------------------------------------------------------------
# Step 3: DoWhy-style Invariant Structural Tests
# ---------------------------------------------------------------------------


def dowhy_refutation(
    signal: FloatArray,
    returns: FloatArray,
    confounders: Optional[FloatArray] = None,
    regime_split: float = 0.5,
    n_bootstrap: int = C.DOWHY_BOOTSTRAP_REPS,
    rng_seed: int = 42,
) -> DoWhyResult:
    """DoWhy placebo treatment + policy invariance refutation.

    Placebo Test (P_VALUE.md §2.A):
        Replace X_t with Z_t ~ N(0, σ²). Compute θ_b for B replications.
        p = (1/B) Σ 𝟏(|θ_b| ≥ |θ_orig|)
        Pass if p > 0.05 (original estimator is distinct from noise).

    Policy Invariance Test (P_VALUE.md §2.B):
        Train on first `regime_split` of data; test on second half.
        p = (1/B) Σ 𝟏(|θ_b - θ_orig| ≥ Δ)  where Δ = expected variance.
        Pass if p > 0.05 (structural coefficient stable across regimes).

    Uses Moving Block Bootstrap to preserve autocorrelation (P_VALUE.md §3).

    Args:
        signal: X (T,).
        returns: Y (T,).
        confounders: Z (T, K) or None.
        regime_split: Fraction of data used as regime 1.
        n_bootstrap: B bootstrap replications.
        rng_seed: Random seed.

    Returns:
        DoWhyResult with γ causal confidence factor.
    """
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    t = len(y)
    rng = np.random.default_rng(rng_seed)

    def _ols_coeff(xv: FloatArray, yv: FloatArray) -> float:
        """OLS slope of yv ~ xv."""
        xv = xv - np.mean(xv)
        denom = float(np.dot(xv, xv))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(xv, yv)) / denom

    theta_orig = _ols_coeff(x, y)

    # ── Placebo Test ─────────────────────────────────────────────────────────
    placebo_thetas: list[float] = []
    sigma_x = float(np.std(x))
    for _ in range(n_bootstrap):
        z_placebo = rng.normal(0.0, sigma_x, size=t)
        placebo_thetas.append(_ols_coeff(z_placebo, y))

    theta_placebo_arr = np.array(placebo_thetas)
    placebo_p = float(np.mean(np.abs(theta_placebo_arr) >= abs(theta_orig)))

    # ── Policy Invariance Test ────────────────────────────────────────────────
    split = max(30, int(t * regime_split))
    theta_r1 = _ols_coeff(x[:split], y[:split])
    theta_r2 = _ols_coeff(x[split:], y[split:])

    # Moving block bootstrap on regime 2 to get Δ threshold
    block_samples = _moving_block_bootstrap(
        np.column_stack([x[split:], y[split:]]),
        block_size=C.BLOCK_BOOTSTRAP_BLOCK_SIZE,
        n_reps=n_bootstrap,
        rng=rng,
    )
    policy_thetas: list[float] = []
    for sample in block_samples:
        if len(sample) < 5:
            continue
        policy_thetas.append(_ols_coeff(sample[:, 0], sample[:, 1]))

    policy_arr = np.array(policy_thetas)
    delta = float(np.std(policy_arr))  # Expected variance threshold
    policy_p = float(
        np.mean(np.abs(policy_arr - theta_r1) >= abs(theta_r2 - theta_r1) + delta)
    )

    placebo_passes = placebo_p < C.CAUSAL_PLACEBO_PVALUE_FLOOR
    policy_passes = policy_p > C.CAUSAL_PLACEBO_PVALUE_FLOOR

    # γ causal confidence factor
    if placebo_passes and policy_passes:
        gamma = C.CAUSAL_GAMMA_HIGH
    elif placebo_passes and not policy_passes:
        gamma = C.CAUSAL_GAMMA_MEDIUM
    else:
        gamma = C.CAUSAL_GAMMA_REJECT

    return DoWhyResult(
        placebo_p_value=placebo_p,
        policy_invariance_p_value=policy_p,
        placebo_passes=placebo_passes,
        policy_passes=policy_passes,
        causal_gamma=gamma,
        n_bootstrap_reps=n_bootstrap,
        theta_orig=theta_orig,
        theta_placebo_mean=float(np.mean(theta_placebo_arr)),
    )


# ---------------------------------------------------------------------------
# Full 3-step causal stack pipeline
# ---------------------------------------------------------------------------


def run_causal_stack(
    signal_name: str,
    signal: FloatArray,
    returns: FloatArray,
    confounders: Optional[FloatArray] = None,
    granger_lags: int = C.GRANGER_LAG_PERIODS,
    n_bootstrap: int = C.DOWHY_BOOTSTRAP_REPS,
    rng_seed: int = 42,
) -> CausalStackResult:
    """Execute the full 3-step causal validation pipeline.

    Step 1 → Granger VARX
    Step 2 → Conditional independence (CMI proxy)
    Step 3 → DoWhy placebo + policy invariance

    γ routing:
      Granger FAILS → REJECT (γ = 0)
      Granger PASS, CMI FAILS (retained < 0.5) → BETA_PROXY (γ = 0.3)
      Granger PASS, CMI PASS, DoWhy FAILS placebo → REJECT (γ = 0)
      All PASS → PASS (γ from DoWhy = 0.95)

    Args:
        signal_name: Identifier string.
        signal: X (T,).
        returns: Y (T,).
        confounders: Z (T, K) or None.
        granger_lags: Max VARX lag order.
        n_bootstrap: DoWhy bootstrap reps.
        rng_seed: Reproducibility seed.

    Returns:
        CausalStackResult with final γ and recommendation.
    """
    granger = granger_causality_varx(
        signal, returns, exogenous=confounders, max_lags=granger_lags
    )

    if not granger.passes:
        cmi = CMIResult(0.0, 1.0, False, 0.0)
        dowhy = DoWhyResult(0.0, 0.0, False, False, C.CAUSAL_GAMMA_REJECT, 0, 0.0, 0.0)
        gamma = C.CAUSAL_GAMMA_REJECT
        rec = "REJECT"
    else:
        cmi = conditional_independence_test(
            signal,
            returns,
            confounders if confounders is not None else np.zeros((len(signal), 1)),
        )
        if not cmi.passes:
            dowhy = DoWhyResult(
                0.0, 0.0, False, False, C.CAUSAL_GAMMA_MEDIUM, 0, 0.0, 0.0
            )
            gamma = C.CAUSAL_GAMMA_MEDIUM
            rec = "BETA_PROXY"
        else:
            dowhy = dowhy_refutation(
                signal,
                returns,
                confounders=confounders,
                n_bootstrap=n_bootstrap,
                rng_seed=rng_seed,
            )
            gamma = dowhy.causal_gamma
            if not dowhy.placebo_passes:
                rec = "REJECT"
            elif not dowhy.policy_passes:
                rec = "BETA_PROXY"
            else:
                rec = "PASS"

    lines = [
        f"Signal: {signal_name}",
        f"  Step 1 — Granger VARX : F={granger.f_statistic:.3f} p={granger.p_value:.4f} "
        f"lag={granger.lag_used} → {'✓ PASS' if granger.passes else '✗ FAIL'}",
        f"  Step 2 — CMI          : stat={cmi.cmi_statistic:.4f} retained={cmi.alpha_retained_fraction:.2%}"
        f" → {'✓ PASS' if cmi.passes else '✗ FAIL'}",
        f"  Step 3 — DoWhy Placebo: p={dowhy.placebo_p_value:.4f} → "
        f"{'✓ PASS' if dowhy.placebo_passes else '✗ FAIL'}",
        f"  Step 3 — Policy Inv.  : p={dowhy.policy_invariance_p_value:.4f} → "
        f"{'✓ PASS' if dowhy.policy_passes else '✗ FAIL'}",
        f"  γ (Causal Confidence) : {gamma:.2f}",
        f"  Recommendation        : {'✓ ' if rec=='PASS' else '⚠ ' if rec=='BETA_PROXY' else '✗ '}{rec}",
    ]

    return CausalStackResult(
        signal_name=signal_name,
        granger=granger,
        cmi=cmi,
        dowhy=dowhy,
        final_gamma=gamma,
        recommendation=rec,
        summary="\n".join(lines),
    )
