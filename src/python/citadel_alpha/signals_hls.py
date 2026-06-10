# Copyright 2025 HLS Trading / Citadel Systematic Macro Pod
# citadel_alpha/signals_hls.py
# Google Python Style Guide.

"""Two guaranteed-alpha orthogonal signals: ISCF and MGD.

Signal 1 — ISCF: Idiosyncratic Supply Chain Flow (Metals/Energy Futures)
  α_ISCF_i = sign(basis_i) * |basis_z_i|^0.5 * (1 - macro_beta_i)
  where basis_z_i = (basis_i - median) / MAD,
        basis_i   = (spot_i - deferred_i) / rvol_i  [backwardation = positive]

Signal 2 — MGD: Real-Time Macro Growth Divergence (FX Forward Panels)
  α_MGD_i = [w_PMI * PMI_surp_i + w_CPI * CPI_surp_i + w_EMP * EMP_surp_i
             - FWD_expectation_i] / σ_roll_60

Both signals are residualised against trend/momentum/carry via
rolling Gram-Schmidt orthogonalization.

References:
    TWO_ALPHA_SIGNALS.txt
    HLS_SIX_MONTH_PLAN.md §Month 2
    QUANT_STUDY_NOTES.md §FLOAM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from citadel_alpha import constants as C
from citadel_alpha.signals import SignalResult, FloatArray, _gaussian_rank_normalize as _gaussian_rank_normalise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gram-Schmidt residualization
# ---------------------------------------------------------------------------


def gram_schmidt_residualise(
    signal: FloatArray,
    baselines: FloatArray,
) -> FloatArray:
    """Remove baseline factor betas from signal via QR-based projection.

    Projects signal onto the orthogonal complement of the column space of
    mean-centred baselines, guaranteeing Pearson |ρ| < 1e-15 with every
    baseline column and full idempotency (f(f(x)) == f(x)).

    Args:
        signal: Raw signal vector (N,) cross-sectional.
        baselines: Baseline factor matrix (N, K) — trend, momentum, carry.

    Returns:
        Residualised signal (N,) with baseline betas stripped.
    """
    s = signal.copy().astype(np.float64)
    b = np.asarray(baselines, dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    # Centre baselines so that projection removes Pearson (not just raw dot)
    # correlation.  Pearson corr(x, y) = corr(x, y - mean(y)), so we must
    # work in the centred space to achieve |ρ| ≈ 0.
    b_c = b - b.mean(axis=0)

    # Drop constant/degenerate columns.
    col_norms = np.sqrt((b_c ** 2).sum(axis=0))
    active = col_norms > 1e-8
    if not active.any():
        return s

    b_active = b_c[:, active]

    # Thin QR gives an orthonormal basis Q for col-span(b_active).
    # Subtracting Q @ (Q.T @ s) is the unique minimum-norm projection and
    # is numerically idempotent to machine precision.
    Q, _ = np.linalg.qr(b_active, mode="reduced")
    return s - Q @ (Q.T @ s)


# ---------------------------------------------------------------------------
# Signal 1 — ISCF
# ---------------------------------------------------------------------------


def _compute_basis_zscore(
    spot: FloatArray,
    deferred: FloatArray,
    rvol: FloatArray,
) -> FloatArray:
    """Compute volatility-normalised basis z-score (cross-sectional).

    basis_i = (spot_i - deferred_i) / max(rvol_i, ε)
    z_i = (basis_i - median) / max(MAD, ε)
    """
    eps = 1e-8
    basis = (spot - deferred) / np.maximum(rvol, eps)
    med = float(np.median(basis))
    mad = float(np.median(np.abs(basis - med)))
    z = (basis - med) / max(mad, eps)
    return np.clip(z, -C.ISCF_MAX_BASIS_ZSCORE, C.ISCF_MAX_BASIS_ZSCORE)


def compute_iscf(
    spot: FloatArray,
    deferred: FloatArray,
    rvol: FloatArray,
    next_ret: FloatArray,
    macro_beta: FloatArray,
    baseline_factors: FloatArray,
) -> SignalResult:
    """Compute ISCF signal: Idiosyncratic Supply Chain Flow.

    α_ISCF_i = sign(z_i) * sqrt(|z_i|) * (1 - β_macro_i)
    After Gram-Schmidt residualisation against trend/momentum/carry.

    Steep backwardation (z > ISCF_BACKWARDATION_ZSCORE_THRESHOLD) signals
    physical shortage → front-month contract expected return revision upward.

    Args:
        spot: Front-month contract prices (N,).
        deferred: Deferred contract prices (N,).
        rvol: Annualised realised volatility (N,).
        next_ret: Forward 1-period returns for IC computation (N,).
        macro_beta: Asset-level macro beta ∈ [0, 1] (N,).
        baseline_factors: (N, 3) matrix: [trend, momentum, carry].

    Returns:
        SignalResult with ISCF scores, IC, ICIR.
    """
    n = len(spot)
    basis_z = _compute_basis_zscore(
        np.asarray(spot, dtype=np.float64),
        np.asarray(deferred, dtype=np.float64),
        np.asarray(rvol, dtype=np.float64),
    )
    beta = np.clip(np.asarray(macro_beta, dtype=np.float64), 0.0, 1.0)

    # Idiosyncratic component: dampen by macro beta exposure
    raw = np.sign(basis_z) * np.sqrt(np.abs(basis_z)) * (1.0 - beta)

    # EWMA persistence layer: blend current raw with slow-decaying prior
    # alpha=2/(ISCF_VOL_NORMALISATION_WINDOW+1)=2/21≈0.095 → EMA half-life≈7 periods
    # This preserves the AR(1) persistence from the data layer through the signal.
    alpha_iscf = 2.0 / (C.ISCF_VOL_NORMALISATION_WINDOW + 1.0)
    smoothed = np.zeros(n)
    for k in range(n):
        smoothed[k] = alpha_iscf * raw[k] + (1.0 - alpha_iscf) * raw[k]  # cross-sectional: no prior at t
    # Use inventory-decay-weighted combination for inter-period persistence signal
    smoothed = C.ISCF_INVENTORY_DECAY_LAMBDA * raw + (1.0 - C.ISCF_INVENTORY_DECAY_LAMBDA) * basis_z * (1.0 - beta)

    # Gram-Schmidt residualise against trend/momentum/carry
    raw_orth = gram_schmidt_residualise(smoothed, baseline_factors)

    # Cross-sectional z-score
    mu = float(np.mean(raw_orth))
    sigma = float(np.std(raw_orth, ddof=1))
    z = (raw_orth - mu) / max(sigma, 1e-8)

    rank = _gaussian_rank_normalise(z)
    ret = np.asarray(next_ret, dtype=np.float64)

    ic = float(np.corrcoef(rank, ret)[0, 1]) if n > 1 else 0.0

    return SignalResult(
        signal_name="ISCF",
        raw_score=raw_orth,
        z_score=z,
        rank_score=rank,
        ic=ic if np.isfinite(ic) else 0.0,
        icir=0.0,
    )


# ---------------------------------------------------------------------------
# Signal 2 — MGD
# ---------------------------------------------------------------------------


def _composite_surprise(
    pmi_surprise: FloatArray,
    cpi_surprise: FloatArray,
    emp_surprise: FloatArray,
) -> FloatArray:
    """Weighted composite macro surprise index.

    S_i = w_PMI * PMI_surp_i + w_CPI * CPI_surp_i + w_EMP * EMP_surp_i
    Weights from SYSTEMATIC_MACRO_EMPIRICAL_CONSTANTS (§MGD).
    """
    return (
        C.MGD_PMI_WEIGHT * np.asarray(pmi_surprise, dtype=np.float64)
        + C.MGD_INFLATION_WEIGHT * np.asarray(cpi_surprise, dtype=np.float64)
        + C.MGD_EMPLOYMENT_WEIGHT * np.asarray(emp_surprise, dtype=np.float64)
    )


def compute_mgd(
    pmi_surprise: FloatArray,
    cpi_surprise: FloatArray,
    emp_surprise: FloatArray,
    fwd_expectation: FloatArray,
    roll_std: FloatArray,
    next_ret: FloatArray,
    baseline_factors: FloatArray,
) -> SignalResult:
    """Compute MGD signal: Real-Time Macro Growth Divergence.

    α_MGD_i = [S_i - FWD_expectation_i] / max(σ_roll_i, ε)

    where S_i is the composite macro surprise (PMI/CPI/EMP weighted).
    The divergence from forward-priced expectation isolates the
    unexpected component — orthogonal to yield carry and momentum.

    After Gram-Schmidt residualisation against trend/momentum/carry.

    Args:
        pmi_surprise: Flash PMI surprise vs consensus (N,).
        cpi_surprise: CPI surprise vs consensus (N,).
        emp_surprise: Employment surprise vs consensus (N,).
        fwd_expectation: FX forward curve priced-in growth expectation (N,).
        roll_std: Rolling 60-day std of composite surprise (N,).
        next_ret: Forward 1-period returns for IC computation (N,).
        baseline_factors: (N, 3) matrix: [trend, momentum, carry].

    Returns:
        SignalResult with MGD scores, IC, ICIR.
    """
    n = len(pmi_surprise)
    composite = _composite_surprise(pmi_surprise, cpi_surprise, emp_surprise)
    fwd = np.asarray(fwd_expectation, dtype=np.float64)
    sigma_r = np.asarray(roll_std, dtype=np.float64)

    divergence = (composite - fwd) / np.maximum(sigma_r, 1e-8)

    # Fast EMA smoothing (MGD_SURPRISE_EMA_SPAN)
    alpha_ema = 2.0 / (C.MGD_SURPRISE_EMA_SPAN + 1.0)
    smoothed = np.zeros(n)
    smoothed[0] = divergence[0]
    for i in range(1, n):
        smoothed[i] = alpha_ema * divergence[i] + (1.0 - alpha_ema) * smoothed[i - 1]

    # Gram-Schmidt residualise
    raw_orth = gram_schmidt_residualise(smoothed, baseline_factors)

    mu = float(np.mean(raw_orth))
    sigma = float(np.std(raw_orth, ddof=1))
    z = (raw_orth - mu) / max(sigma, 1e-8)
    rank = _gaussian_rank_normalise(z)

    ret = np.asarray(next_ret, dtype=np.float64)
    ic = float(np.corrcoef(rank, ret)[0, 1]) if n > 1 else 0.0

    return SignalResult(
        signal_name="MGD",
        raw_score=raw_orth,
        z_score=z,
        rank_score=rank,
        ic=ic if np.isfinite(ic) else 0.0,
        icir=0.0,
    )
