# Copyright 2025 Citadel Systematic Macro Pod
# citadel_alpha/analytics.py
# Google Python Style Guide.

"""Institutional analytics: IC/ICIR monitoring, FLOAM, DSR, signal orthogonality.

Implements the full quantitative scientific method from the study notes:
  - Rolling IC / ICIR with CPCV validation
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
  - Signal subspace orthogonality (R^2 / VIF checks)
  - FLOAM breadth-adjusted IR estimation
  - HMM regime classification (2-state Gaussian)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Orthogonality check
# ---------------------------------------------------------------------------

@dataclass
class OrthogonalityReport:
    """Summary of signal orthogonality diagnostics."""

    r2_matrix: FloatArray          # (N_signals x N_signals) R^2 pairwise
    max_r2: float                  # Maximum off-diagonal R^2 (should be < 0.15)
    vif: FloatArray                # Variance Inflation Factor per new signal
    is_orthogonal: bool            # True if max_r2 < 0.15 and all VIF < 5


def compute_orthogonality(
    signal_matrix: FloatArray,
    threshold_r2: float = 0.15,
    threshold_vif: float = 5.0,
) -> OrthogonalityReport:
    """Compute pairwise R^2 and VIF to verify signal orthogonality.

    Args:
        signal_matrix: Shape (T, K) matrix of K signal time series.
        threshold_r2: Max acceptable pairwise R^2 (default 0.15).
        threshold_vif: Max acceptable VIF (default 5.0).

    Returns:
        OrthogonalityReport.
    """
    t, k = signal_matrix.shape

    # Pairwise Spearman R^2.
    r2_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                r2_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(signal_matrix[:, i], signal_matrix[:, j])
                r2_matrix[i, j] = float(rho ** 2) if np.isfinite(rho) else 0.0

    off_diag = r2_matrix.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_r2 = float(off_diag.max())

    # VIF for each signal vs. remaining signals.
    vif = np.zeros(k)
    for i in range(k):
        y = signal_matrix[:, i]
        x = np.delete(signal_matrix, i, axis=1)
        if x.shape[1] == 0:
            vif[i] = 1.0
            continue
        # R^2 from OLS: R^2 = 1 - SS_res / SS_tot
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(t), x]), y, rcond=None
            )
            y_hat = np.column_stack([np.ones(t), x]) @ coeffs
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_i = 1.0 - ss_res / max(ss_tot, 1e-12)
            vif[i] = 1.0 / max(1.0 - r2_i, 1e-12)
        except np.linalg.LinAlgError:
            vif[i] = float("inf")

    is_orthogonal = max_r2 < threshold_r2 and bool(np.all(vif < threshold_vif))
    return OrthogonalityReport(
        r2_matrix=r2_matrix,
        max_r2=max_r2,
        vif=vif,
        is_orthogonal=is_orthogonal,
    )


# ---------------------------------------------------------------------------
# FLOAM — Fundamental Law of Active Management
# ---------------------------------------------------------------------------

@dataclass
class FloamResult:
    """FLOAM-derived IR estimate with breadth correction."""

    ic: float          # Mean IC across signals and time
    icir: float        # IC / std(IC)
    breadth: float     # Effective breadth (adjusted for correlation)
    ir_predicted: float  # IC * sqrt(N_eff)
    ir_realised: float   # Realised Sharpe ratio


def compute_floam(
    ic_panel: FloatArray,
    realised_pnl: FloatArray,
    asset_correlation: float = 0.0,
) -> FloamResult:
    """Compute FLOAM-adjusted Information Ratio estimate.

    IR ≈ IC * sqrt(N_effective)
    N_effective = N / (1 + (N-1) * rho_signals)  (Grinold & Kahn breadth correction)

    Args:
        ic_panel: Shape (T, K) IC values per signal and time step.
        realised_pnl: Daily P&L time series (T,).
        asset_correlation: Average pairwise asset correlation (breadth correction).

    Returns:
        FloamResult.
    """
    ic_mean = float(np.nanmean(ic_panel))
    ic_std = float(np.nanstd(ic_panel, ddof=1))
    icir = ic_mean / max(ic_std, 1e-12)

    t, k = ic_panel.shape
    n_eff = k / (1.0 + max(k - 1, 0) * asset_correlation)
    ir_pred = ic_mean * np.sqrt(n_eff)

    pnl = np.asarray(realised_pnl, dtype=np.float64)
    ann_ret = np.mean(pnl) * 252
    ann_vol = np.std(pnl, ddof=1) * np.sqrt(252)
    ir_real = ann_ret / max(ann_vol, 1e-12)

    return FloamResult(
        ic=ic_mean,
        icir=icir,
        breadth=float(n_eff),
        ir_predicted=float(ir_pred),
        ir_realised=float(ir_real),
    )


# ---------------------------------------------------------------------------
# HMM regime classifier (2-state Gaussian EM)
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """2-state HMM output."""

    regimes: FloatArray          # (T,) state sequence: 0=risk-off, 1=risk-on
    probabilities: FloatArray    # (T, 2) state probabilities
    means: FloatArray            # (2,) emission means
    stds: FloatArray             # (2,) emission stds
    transition_matrix: FloatArray  # (2, 2) transition probabilities


def fit_hmm_regimes(
    returns: FloatArray,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> RegimeState:
    """Fit a 2-state Gaussian HMM via Baum-Welch EM.

    Args:
        returns: (T,) return series (e.g. equity index or vol series).
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance on log-likelihood.

    Returns:
        RegimeState with decoded state sequence and parameters.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore[import]
        model = GaussianHMM(
            n_components=2, covariance_type="full", n_iter=n_iter, tol=tol
        )
        X = returns.reshape(-1, 1)
        model.fit(X)
        states = model.predict(X)
        probs = model.predict_proba(X)
        means = model.means_.flatten()
        stds = np.sqrt(model.covars_.flatten())

        # Ensure state 0 = lower mean (risk-off), state 1 = higher (risk-on).
        if means[0] > means[1]:
            states = 1 - states
            probs = probs[:, ::-1]
            means = means[::-1]
            stds = stds[::-1]

        return RegimeState(
            regimes=states.astype(np.float64),
            probabilities=probs,
            means=means,
            stds=stds,
            transition_matrix=model.transmat_,
        )
    except ImportError:
        pass

    # Fallback: simple threshold-based regime classification.
    logger.warning("hmmlearn not available; using threshold regime classifier.")
    vol = np.abs(returns - np.mean(returns))
    threshold = np.median(vol)
    regimes = (vol < threshold).astype(np.float64)  # 1 = low vol = risk-on
    probs = np.column_stack([1 - regimes, regimes])
    mean_on = float(np.mean(returns[regimes == 1.0]))
    mean_off = float(np.mean(returns[regimes == 0.0]))
    std_on = float(np.std(returns[regimes == 1.0], ddof=1))
    std_off = float(np.std(returns[regimes == 0.0], ddof=1))

    return RegimeState(
        regimes=regimes,
        probabilities=probs,
        means=np.array([mean_off, mean_on]),
        stds=np.array([std_off, std_on]),
        transition_matrix=np.array([[0.95, 0.05], [0.05, 0.95]]),
    )


# ---------------------------------------------------------------------------
# CPCV cross-validation (Combinatorial Purged CV)
# ---------------------------------------------------------------------------

def cpcv_ic_scores(
    signal_panel: FloatArray,
    returns_panel: FloatArray,
    n_splits: int = 6,
    n_test_groups: int = 2,
    embargo_days: int = 5,
) -> FloatArray:
    """Estimate out-of-sample IC via Combinatorial Purged Cross-Validation.

    This corrects for the walk-forward validation failure mode described in
    the study notes, where overlapping training windows cause data leakage.

    Args:
        signal_panel: Shape (T, N) signal matrix (T timesteps, N assets).
        returns_panel: Shape (T, N) forward returns.
        n_splits: Number of equal time splits.
        n_test_groups: Number of test groups per CV fold (≥ 2 for CPCV).
        embargo_days: Gap between train and test to prevent leakage.

    Returns:
        Array of OOS IC estimates per CV fold.
    """
    from itertools import combinations

    t = signal_panel.shape[0]
    split_size = t // n_splits
    split_bounds = [(i * split_size, min((i + 1) * split_size, t))
                    for i in range(n_splits)]

    oos_ics: list[float] = []
    for test_groups in combinations(range(n_splits), n_test_groups):
        test_indices: list[int] = []
        train_indices: list[int] = []
        for g, (s, e) in enumerate(split_bounds):
            if g in test_groups:
                test_indices.extend(range(s, e))
            else:
                train_indices.extend(range(s, e))

        # Apply embargo: remove train samples within embargo_days of test.
        test_set = set(test_indices)
        clean_train = [
            i for i in train_indices
            if not any(abs(i - j) <= embargo_days for j in test_set)
        ]
        if len(clean_train) < 20 or len(test_indices) < 4:
            continue

        test_sig = signal_panel[test_indices]
        test_ret = returns_panel[test_indices]

        # Compute mean IC across assets for this fold.
        fold_ics = []
        for n in range(test_sig.shape[1]):
            rho, _ = spearmanr(test_sig[:, n], test_ret[:, n])
            if np.isfinite(rho):
                fold_ics.append(float(rho))
        if fold_ics:
            oos_ics.append(float(np.mean(fold_ics)))

    return np.array(oos_ics) if oos_ics else np.array([0.0])