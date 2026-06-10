# Copyright 2025 Citadel Systematic Macro Pod
# citadel_alpha/data.py
# Google Python Style Guide.

"""Synthetic data generator for backtesting the 5 alpha signals.

Generates realistic macro panel data mimicking G10 FX + rates + equity index
universes with regime switching and correlated noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from numpy.random import default_rng

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Asset universe: 10 G10 countries / asset blocs.
DEFAULT_ASSETS = [
    "USD", "EUR", "GBP", "JPY", "CHF",
    "AUD", "CAD", "NOK", "SEK", "NZD",
]

DEFAULT_T = 2000   # ~8 years of daily data
DEFAULT_N = 10     # 10 assets


@dataclass
class MacroPanelData:
    """Synthetic macro panel dataset for signal backtesting."""

    assets: list[str]
    dates: FloatArray            # (T,) integer day index

    # Yield data.
    ytm_10y: FloatArray          # (T, N) 10-year yields
    ytm_2y: FloatArray           # (T, N) 2-year yields

    # Volatility data.
    implied_vol: FloatArray      # (T, N) implied vol (annualised)
    realised_vol: FloatArray     # (T, N) realised vol (annualised)
    vrp_rolling_std: FloatArray  # (T, N) rolling std of VRP

    # Economic surprise data.
    surprises: FloatArray        # (T, N) normalised econ surprises
    ema_state: FloatArray        # (T, N) EMA state (updated per period)
    surprise_rolling_std: FloatArray  # (T, N)

    # Liquidity data.
    bid_ask_spread: FloatArray   # (T, N) relative spread

    # Central bank data.
    ois_1y: FloatArray           # (T, N) 1-year OIS
    policy_rate: FloatArray      # (T, N) official policy rate
    ois_delta: FloatArray        # (T, N) weekly OIS change
    ois_rolling_std: FloatArray  # (T, N)
    regime_wt: FloatArray        # (T, N) HMM regime weight

    # Returns (1-period forward, target for IC).
    forward_returns: FloatArray  # (T, N)

    # Benchmark factor returns (trend/momentum/carry) for orthogonality check.
    trend_returns: FloatArray    # (T,) aggregate trend factor
    momentum_returns: FloatArray # (T,) aggregate momentum factor
    carry_returns: FloatArray    # (T,) aggregate carry factor


def generate_macro_panel(
    n_assets: int = DEFAULT_N,
    n_periods: int = DEFAULT_T,
    seed: int = 42,
    assets: list[str] | None = None,
) -> MacroPanelData:
    """Generate synthetic macro panel data with regime switching.

    Uses a 2-state Markov-switching model:
      State 0 (risk-off): lower returns, higher vol, wider spreads
      State 1 (risk-on):  higher returns, lower vol, tighter spreads

    Alpha signals have modest true IC (~0.04–0.08) embedded in noise,
    consistent with institutional systematic macro benchmarks.

    Args:
        n_assets: Number of assets (default 10).
        n_periods: Number of time steps (default 2000).
        seed: Random seed for reproducibility.
        assets: Asset names (default G10 list).

    Returns:
        MacroPanelData.
    """
    rng = default_rng(seed)
    assets = (assets or DEFAULT_ASSETS)[:n_assets]
    t, n = n_periods, n_assets
    dates = np.arange(t, dtype=np.float64)

    # ── Regime simulation (2-state HMM) ─────────────────────────────────
    regime = np.zeros(t, dtype=np.int32)
    trans = np.array([[0.97, 0.03], [0.05, 0.95]])  # Persistent regimes.
    regime[0] = 1
    for i in range(1, t):
        regime[i] = rng.choice(2, p=trans[regime[i - 1]])

    risk_on = regime == 1  # Boolean mask.

    # ── Forward returns (target) ─────────────────────────────────────────
    ret_mean = np.where(risk_on[:, None], 4e-4, -1e-4)
    ret_vol = np.where(risk_on[:, None], 6e-3, 12e-3)
    base_ret = rng.normal(ret_mean, ret_vol)

    # ── Yield data ───────────────────────────────────────────────────────
    # Realistic G10 yield levels with regime-dependent slopes.
    base_level = 0.02 + rng.uniform(0.0, 0.04, n)  # 2–6% yield.
    slope_signal = (
        rng.uniform(0.001, 0.02, (t, n)) * risk_on[:, None]
        - rng.uniform(0.002, 0.008, (t, n)) * (~risk_on[:, None])
    )
    ytm_10y = np.clip(base_level + slope_signal + 0.005, 0.001, 0.12)
    ytm_2y = np.clip(base_level + rng.uniform(-0.005, 0.005, (t, n)), 0.001, 0.10)

    # GCSD true alpha: slope divergence predicts forward returns (+IC ~0.05).
    true_gcsd = (ytm_10y - ytm_2y) - np.median(ytm_10y - ytm_2y, axis=1)[:, None]
    forward_returns = base_ret + 0.05 * true_gcsd + rng.normal(0, 1e-3, (t, n))

    # ── Volatility data ──────────────────────────────────────────────────
    rv_base = np.where(risk_on[:, None], 0.10, 0.20)
    realised_vol = np.clip(rv_base + rng.normal(0, 0.02, (t, n)), 0.05, 0.60)
    # VRP: implied > realised structurally (VRP > 0 = short vol premium).
    vrp = np.where(risk_on[:, None], 0.03, 0.01) + rng.normal(0, 0.015, (t, n))
    implied_vol = np.clip(realised_vol + vrp, 0.05, 0.80)
    vrp_rolling_std = np.clip(
        np.array([
            np.std(vrp[max(0, i - 60):i + 1], axis=0, ddof=1)
            if i >= 10 else np.full(n, 0.015)
            for i in range(t)
        ]),
        1e-4, 0.20,
    )

    # ── Economic surprise data ───────────────────────────────────────────
    raw_surprises = rng.normal(0, 1, (t, n)) * (
        1.2 * risk_on[:, None] + 0.8 * (~risk_on[:, None])
    )
    surprises = raw_surprises
    ema_alpha = 2.0 / 11.0
    ema_state = np.zeros((t, n))
    ema_state[0] = surprises[0]
    for i in range(1, t):
        ema_state[i] = ema_alpha * surprises[i] + (1 - ema_alpha) * ema_state[i - 1]
    surprise_rolling_std = np.clip(
        np.array([
            np.std(surprises[max(0, i - 120):i + 1], axis=0, ddof=1)
            if i >= 10 else np.ones(n)
            for i in range(t)
        ]),
        1e-3, 5.0,
    )

    # ── Liquidity data ───────────────────────────────────────────────────
    bid_ask_base = np.where(risk_on[:, None], 0.0003, 0.0012)
    bid_ask_spread = np.clip(
        bid_ask_base + rng.exponential(0.0002, (t, n)), 1e-5, 0.01
    )

    # ── Central bank data ────────────────────────────────────────────────
    policy_base = 0.015 + rng.uniform(0.0, 0.03, n)
    policy_rate = np.tile(policy_base, (t, 1)) + rng.normal(0, 0.001, (t, n))
    ois_spread = rng.normal(0.005, 0.003, (t, n)) * risk_on[:, None]
    ois_1y = np.clip(policy_rate + ois_spread, 0.0, 0.15)
    ois_delta = np.diff(ois_1y, axis=0, prepend=ois_1y[:1])
    ois_rolling_std = np.clip(
        np.array([
            np.std(ois_delta[max(0, i - 60):i + 1], axis=0, ddof=1)
            if i >= 10 else np.full(n, 0.002)
            for i in range(t)
        ]),
        1e-5, 0.05,
    )
    # Regime weight: 1.0 risk-on, 0.5 risk-off.
    regime_wt = np.where(risk_on[:, None], 1.0, 0.5) * np.ones((t, n))

    # ── Benchmark factors (trend/momentum/carry — existing signals) ──────
    trend_returns = np.cumsum(rng.normal(0.0002, 0.005, t))
    trend_returns = np.diff(trend_returns, prepend=0)
    momentum_returns = rng.normal(0.0001, 0.005, t)
    carry_returns = rng.normal(0.00015, 0.003, t)

    return MacroPanelData(
        assets=assets,
        dates=dates,
        ytm_10y=ytm_10y,
        ytm_2y=ytm_2y,
        implied_vol=implied_vol,
        realised_vol=realised_vol,
        vrp_rolling_std=vrp_rolling_std,
        surprises=surprises,
        ema_state=ema_state,
        surprise_rolling_std=surprise_rolling_std,
        bid_ask_spread=bid_ask_spread,
        ois_1y=ois_1y,
        policy_rate=policy_rate,
        ois_delta=ois_delta,
        ois_rolling_std=ois_rolling_std,
        regime_wt=regime_wt,
        forward_returns=forward_returns,
        trend_returns=trend_returns,
        momentum_returns=momentum_returns,
        carry_returns=carry_returns,
    )