# citadel_alpha/signals.py — Proprietary Trading Firm — Alpha Engine signal utilities.
# Google Python Style Guide.

"""Signal utilities and base classes for Proprietary Trading Firm — alpha engine.

Exports used by signals_hls.py (ISCF, MGD) and the causal framework.
The legacy signals have been removed;
this branch focuses exclusively on ISCF and MGD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from citadel_alpha import constants as C

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

try:
    from citadel_alpha import _citadel_alpha_cpp as _cpp  # type: ignore[import]

    _CPP_AVAILABLE = True
    logger.info("C++ alpha engine loaded.")
except ImportError:
    _cpp = None  # type: ignore[assignment]
    _CPP_AVAILABLE = False
    logger.warning("C++ extension not found; using pure-Python fallback.")


@dataclass
class SignalResult:
    """Per-asset signal output with metadata for IC/ICIR monitoring."""

    signal_name: str
    raw_score: FloatArray
    z_score: FloatArray
    rank_score: FloatArray
    ic: float = 0.0
    icir: float = 0.0

    def __post_init__(self) -> None:
        self.raw_score = np.asarray(self.raw_score, dtype=np.float64)
        self.z_score = np.asarray(self.z_score, dtype=np.float64)
        self.rank_score = np.asarray(self.rank_score, dtype=np.float64)


def _cross_sectional_zscore(x: FloatArray) -> FloatArray:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    return (x - mu) / max(sigma, 1e-8)


def _robust_zscore(x: FloatArray) -> FloatArray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return (x - med) / max(mad, 1e-8)


def _gaussian_rank_normalize(x: FloatArray) -> FloatArray:
    n = len(x)
    order = np.argsort(x)
    rank = np.empty(n)
    for r, idx in enumerate(order):
        u = (r + 1.0) / (n + 1.0)
        rank[idx] = float(stats.norm.ppf(u))
    return rank


def _compute_ic(signal: FloatArray, returns: FloatArray) -> float:
    if len(signal) < 4:
        return 0.0
    corr = float(np.corrcoef(signal, returns)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def rolling_icir(ic_series: FloatArray, window: int = 60) -> FloatArray:
    """Rolling ICIR = rolling_mean(IC) / rolling_std(IC).

    Returns NaN for the first ``window - 1`` elements (warm-up period).
    """
    ic = np.asarray(ic_series, dtype=np.float64)
    t = len(ic)
    result = np.full(t, np.nan)
    for i in range(window - 1, t):
        chunk = ic[i - window + 1 : i + 1]
        mu = float(np.mean(chunk))
        sigma = float(np.std(chunk, ddof=1))
        result[i] = mu / max(sigma, 1e-8)
    return result


def vix_regime_scale(vix_level: float) -> float:
    """VIX-based signal scaling factor (§5.8 constants)."""
    if vix_level < C.VIX_RISK_ON_THRESHOLD:
        return C.VIX_RISK_ON_SCALE
    if vix_level < C.VIX_CRISIS_THRESHOLD:
        return C.VIX_TRANSITION_SCALE
    return C.VIX_CRISIS_SCALE


# ---------------------------------------------------------------------------
# Signal compute functions (legacy 5-signal suite)
# ---------------------------------------------------------------------------


def _validate_min_n(arr: FloatArray, name: str = "array", min_n: int = 4) -> None:
    if len(arr) < min_n:
        raise ValueError(f"{name} must have at least {min_n} elements, got {len(arr)}")


def _validate_same_size(*arrays: FloatArray) -> None:
    sizes = [len(a) for a in arrays]
    if len(set(sizes)) > 1:
        raise ValueError(f"All arrays must have the same size, got sizes: {sizes}")


def compute_gcsd(
    ytm_10y: FloatArray,
    ytm_2y: FloatArray,
    next_ret: FloatArray,
    vix_level: float = 20.0,
    use_cpp: bool = True,
) -> SignalResult:
    """Global Yield Curve Slope Divergence."""
    ytm_10y = np.asarray(ytm_10y, dtype=np.float64)
    ytm_2y = np.asarray(ytm_2y, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    _validate_min_n(ytm_10y, "ytm_10y")
    _validate_same_size(ytm_10y, ytm_2y, next_ret)

    if use_cpp and _CPP_AVAILABLE:
        try:
            result = _cpp.GlobalYieldCurveSlopeDivergence.compute(
                ytm_10y, ytm_2y, next_ret
            )
            raw = np.asarray(result.raw_score)
            z = np.asarray(result.z_score)
            rank = np.asarray(result.rank_score)
            scale = vix_regime_scale(vix_level)
            return SignalResult(
                "GCSD", raw * scale, z, rank, float(result.ic), float(result.icir)
            )
        except Exception:
            pass

    slope = np.log(ytm_10y) - np.log(ytm_2y)
    med = float(np.median(slope))
    mad = float(np.median(np.abs(slope - med)))
    raw = (slope - med) / max(mad, 1e-8)
    scale = vix_regime_scale(vix_level)
    raw = raw * scale
    z = _cross_sectional_zscore(raw)
    rank = _gaussian_rank_normalize(raw)
    ic = _compute_ic(rank, next_ret)
    return SignalResult("GCSD", raw, z, rank, ic)


def compute_rvis(
    iv_current: FloatArray,
    rv_current: FloatArray,
    vrp_rolling_std: FloatArray,
    next_ret: FloatArray,
    use_cpp: bool = True,
) -> SignalResult:
    """Realised vs Implied Volatility Spread Cross-Asset."""
    iv_current = np.asarray(iv_current, dtype=np.float64)
    rv_current = np.asarray(rv_current, dtype=np.float64)
    vrp_rolling_std = np.asarray(vrp_rolling_std, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    _validate_min_n(iv_current, "iv_current")
    _validate_same_size(iv_current, rv_current, vrp_rolling_std, next_ret)

    if use_cpp and _CPP_AVAILABLE:
        try:
            result = _cpp.RealisedImpliedVolSpread.compute(
                iv_current, rv_current, vrp_rolling_std, next_ret
            )
            return SignalResult(
                "RVIS",
                np.asarray(result.raw_score),
                np.asarray(result.z_score),
                np.asarray(result.rank_score),
                float(result.ic),
                float(result.icir),
            )
        except Exception:
            pass

    vrp = iv_current - rv_current
    raw = vrp / np.maximum(vrp_rolling_std, 1e-8)
    z = _cross_sectional_zscore(raw)
    rank = _gaussian_rank_normalize(raw)
    ic = _compute_ic(rank, next_ret)
    return SignalResult("RVIS", raw, z, rank, ic)


def compute_msdi(
    surprises: FloatArray,
    ema_prev: FloatArray,
    rolling_std: FloatArray,
    next_ret: FloatArray,
    ema_window: int = 10,
    use_cpp: bool = True,
) -> SignalResult:
    """Macro Surprise Diffusion Index."""
    surprises = np.asarray(surprises, dtype=np.float64)
    ema_prev = np.asarray(ema_prev, dtype=np.float64)
    rolling_std = np.asarray(rolling_std, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    _validate_min_n(surprises, "surprises")
    _validate_same_size(surprises, ema_prev, rolling_std, next_ret)

    ema_alpha = 2.0 / (ema_window + 1.0)
    ema_val = ema_alpha * surprises + (1.0 - ema_alpha) * ema_prev
    raw = ema_val / np.maximum(rolling_std, 1e-8)
    z = _cross_sectional_zscore(raw)
    rank = _gaussian_rank_normalize(raw)
    ic = _compute_ic(rank, next_ret)
    return SignalResult("MSDI", raw, z, rank, ic)


def compute_clsd(
    bid_ask_spread: FloatArray,
    next_ret: FloatArray,
    use_cpp: bool = True,
) -> SignalResult:
    """Cross-Asset Liquidity Stress Divergence."""
    bid_ask_spread = np.asarray(bid_ask_spread, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    _validate_min_n(bid_ask_spread, "bid_ask_spread")
    _validate_same_size(bid_ask_spread, next_ret)

    med = float(np.median(bid_ask_spread))
    mad = float(np.median(np.abs(bid_ask_spread - med)))
    liq_z = (bid_ask_spread - med) / max(mad, 1e-8)
    raw = -np.sign(liq_z) * np.sqrt(np.abs(liq_z))
    z = _cross_sectional_zscore(raw)
    rank = _gaussian_rank_normalize(raw)
    ic = _compute_ic(rank, next_ret)
    return SignalResult("CLSD", raw, z, rank, ic)


def compute_cbpsm(
    ois_1y: FloatArray,
    policy_rate: FloatArray,
    ois_delta: FloatArray,
    rolling_std: FloatArray,
    regime_wt: FloatArray,
    next_ret: FloatArray,
    use_cpp: bool = True,
) -> SignalResult:
    """Central Bank Policy Surprise Momentum."""
    ois_1y = np.asarray(ois_1y, dtype=np.float64)
    policy_rate = np.asarray(policy_rate, dtype=np.float64)
    ois_delta = np.asarray(ois_delta, dtype=np.float64)
    rolling_std = np.asarray(rolling_std, dtype=np.float64)
    regime_wt = np.clip(np.asarray(regime_wt, dtype=np.float64), 0.5, 1.0)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    _validate_min_n(ois_1y, "ois_1y")
    _validate_same_size(
        ois_1y, policy_rate, ois_delta, rolling_std, regime_wt, next_ret
    )

    raw = (ois_delta / np.maximum(rolling_std, 1e-8)) * regime_wt
    z = _cross_sectional_zscore(raw)
    rank = _gaussian_rank_normalize(raw)
    ic = _compute_ic(rank, next_ret)
    return SignalResult("CBPSM", raw, z, rank, ic)
