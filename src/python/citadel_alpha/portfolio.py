# Copyright 2025 Citadel Systematic Macro Pod
# citadel_alpha/portfolio.py
# Google Python Style Guide.

"""Portfolio construction: HRP + Fractional Kelly + Ledoit-Wolf shrinkage.

Follows data-oriented design: all intermediate results returned as
contiguous NumPy arrays for cache efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Hard constraints (institutional production limits).
MAX_SIGNAL_WEIGHT = 0.15  # 15% max risk cap per signal (per study notes).
MIN_KELLY_FRACTION = 0.25  # Conservative fractional Kelly floor.
MAX_KELLY_FRACTION = 0.50  # Standard institutional half-Kelly cap.


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage (analytical Oracle Approximating Shrinkage)
# ---------------------------------------------------------------------------

def ledoit_wolf_shrink(returns_matrix: FloatArray) -> FloatArray:
    """Compute Ledoit-Wolf shrunk covariance matrix.

    Uses the Oracle Approximating Shrinkage (OAS) estimator which is
    analytically optimal for finite T, N without cross-validation.

    Args:
        returns_matrix: Shape (T, N) returns matrix.

    Returns:
        Shape (N, N) shrunk covariance matrix.
    """
    try:
        from sklearn.covariance import OAS  # type: ignore[import]
        oas = OAS()
        oas.fit(returns_matrix)
        return oas.covariance_
    except ImportError:
        pass

    # Manual analytical Ledoit-Wolf (constant-correlation target).
    t, n = returns_matrix.shape
    s = np.cov(returns_matrix, rowvar=False)
    trace_s = np.trace(s)
    trace_s2 = np.trace(s @ s)
    frob_s = np.sum(s * s)

    mu = trace_s / n
    delta_num = frob_s + trace_s ** 2
    delta_den = (t + 1 - 2.0 / n) * (frob_s - trace_s2 / n)
    delta = max(0.0, min(1.0, delta_num / max(delta_den, 1e-12)))

    target = mu * np.eye(n)
    return delta * target + (1.0 - delta) * s


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP)
# ---------------------------------------------------------------------------

def _get_cluster_variance(cov: FloatArray, cluster_items: list[int]) -> float:
    """Inverse-variance weight cluster variance."""
    sub_cov = cov[np.ix_(cluster_items, cluster_items)]
    ivp = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def hierarchical_risk_parity(
    returns_matrix: FloatArray,
    signal_scores: FloatArray,
) -> FloatArray:
    """Compute HRP weights incorporating signal score tilts.

    Args:
        returns_matrix: Shape (T, N) historical returns.
        signal_scores: Shape (N,) composite signal z-scores for tilt.

    Returns:
        Shape (N,) portfolio weights (long-short, sum to 0).
    """
    t, n = returns_matrix.shape
    if n < 2:
        return signal_scores / np.maximum(np.abs(signal_scores).sum(), 1e-12)

    # Step 1: Ledoit-Wolf covariance.
    cov = ledoit_wolf_shrink(returns_matrix)

    # Step 2: Correlation-based distance.
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    dist_condensed = squareform(dist, checks=False)

    # Step 3: Hierarchical clustering (Ward linkage).
    linkage = hierarchy.linkage(dist_condensed, method="ward")

    # Step 4: Recursive bisection quasi-diagonalisation.
    sorted_idx = list(hierarchy.leaves_list(linkage))
    weights = np.ones(n)
    cluster_list: list[list[int]] = [sorted_idx]

    while cluster_list:
        current = cluster_list.pop()
        if len(current) <= 1:
            continue
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]

        var_left = _get_cluster_variance(cov, left)
        var_right = _get_cluster_variance(cov, right)
        alpha = 1.0 - var_left / max(var_left + var_right, 1e-12)

        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        cluster_list.extend([left, right])

    # Step 5: Apply signal score tilt (long-short construction).
    # Map weights to long-short: positive signal → long, negative → short.
    signal_norm = signal_scores / np.maximum(np.abs(signal_scores).max(), 1e-12)
    hrp_weights = weights * signal_norm  # Element-wise tilt.

    # Enforce 15% max weight cap (institutional limit per study notes).
    max_abs = np.abs(hrp_weights).max()
    if max_abs > MAX_SIGNAL_WEIGHT:
        hrp_weights *= MAX_SIGNAL_WEIGHT / max_abs

    return hrp_weights


# ---------------------------------------------------------------------------
# Fractional Kelly sizing
# ---------------------------------------------------------------------------

def fractional_kelly_sizing(
    expected_alpha: FloatArray,
    variance: FloatArray,
    fraction: float = MAX_KELLY_FRACTION,
) -> FloatArray:
    """Compute fractional Kelly position sizes.

    f_i = fraction * (E[alpha_i] / Var[alpha_i])

    Args:
        expected_alpha: Expected alpha per asset (N,).
        variance: Variance of alpha per asset (N,).
        fraction: Kelly fraction [0.25, 0.5] for institutional conservatism.

    Returns:
        Position sizes (N,), clipped to [-MAX_SIGNAL_WEIGHT, MAX_SIGNAL_WEIGHT].
    """
    fraction = float(np.clip(fraction, MIN_KELLY_FRACTION, MAX_KELLY_FRACTION))
    var_safe = np.maximum(variance, 1e-12)
    kelly = fraction * (expected_alpha / var_safe)
    return np.clip(kelly, -MAX_SIGNAL_WEIGHT, MAX_SIGNAL_WEIGHT)


# ---------------------------------------------------------------------------
# Portfolio performance metrics
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMetrics:
    """Key portfolio performance metrics."""

    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    annualised_return: float
    annualised_vol: float
    hit_rate: float
    avg_ic: float
    avg_icir: float


def compute_portfolio_metrics(
    pnl_series: FloatArray,
    ic_series: FloatArray,
    icir_series: FloatArray,
    periods_per_year: int = 252,
) -> PortfolioMetrics:
    """Compute institutional-grade portfolio performance metrics.

    Args:
        pnl_series: Daily P&L time series.
        ic_series: Daily IC values per signal (averaged if multi-signal).
        icir_series: Daily ICIR values.
        periods_per_year: Annualisation factor (252 for daily).

    Returns:
        PortfolioMetrics dataclass.
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    scale = np.sqrt(periods_per_year)

    ann_ret = np.mean(pnl) * periods_per_year
    ann_vol = np.std(pnl, ddof=1) * scale
    sharpe = ann_ret / max(ann_vol, 1e-12)

    downside = pnl[pnl < 0.0]
    sortino_denom = np.std(downside, ddof=1) * scale if len(downside) > 1 else 1e-12
    sortino = ann_ret / max(sortino_denom, 1e-12)

    # Maximum drawdown.
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(np.min(drawdown))

    calmar = ann_ret / max(abs(max_dd), 1e-12)
    hit_rate = float(np.mean(pnl > 0.0))

    return PortfolioMetrics(
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        calmar=float(calmar),
        annualised_return=float(ann_ret),
        annualised_vol=float(ann_vol),
        hit_rate=float(hit_rate),
        avg_ic=float(np.nanmean(ic_series)),
        avg_icir=float(np.nanmean(icir_series)),
    )


def deflated_sharpe_ratio(
    sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
) -> float:
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts for selection bias from multiple testing over N_trials.

    Args:
        sr: Estimated Sharpe ratio of the strategy.
        sr_benchmark: Expected maximum Sharpe from random search (N_trials).
        n_obs: Number of observations (T).
        skew: Return distribution skewness.
        kurt: Return distribution excess kurtosis.
        n_trials: Number of strategies trialled before selection.

    Returns:
        DSR ∈ [0, 1]: probability that SR > 0 after multiple-test deflation.
    """
    from scipy.stats import norm

    if sr_benchmark <= 0.0:
        sr_benchmark = np.sqrt(2.0 * np.log(n_trials)) / np.sqrt(
            np.log(n_trials * np.log(n_trials + 1e-6))
        )

    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    denom = max(denom, 1e-12)
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))