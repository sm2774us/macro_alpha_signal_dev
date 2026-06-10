# Copyright 2025 Citadel Systematic Macro Pod
# citadel_alpha/constants.py
# Google Python Style Guide.

"""Empirical constants for Citadel Systematic Macro Alpha Engine.

All constants calibrated for liquid macro markets (G10 FX, Rates, Equity
Index, Commodities). Based on SYSTEMATIC_MACRO_EMPIRICAL_CONSTANTS.md.

References:
    SYSTEMATIC_MACRO_EMPIRICAL_CONSTANTS.md — §1-§5.8
    QUANT_STUDY_NOTES.md — §0.4 Statistical Testing Framework
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# §1 Mean-Reversion & Decay Rates
# ---------------------------------------------------------------------------

OU_KAPPA: float = 0.01
"""Ornstein-Uhlenbeck mean-reversion rate (day^-1). Half-life ≈ 69 days."""

ALPHA_HALF_LIFE_MIN: int = 21
"""Minimum alpha decay half-life (days). Below this: retire signal."""

ALPHA_HALF_LIFE_MAX: int = 63
"""Maximum alpha decay half-life (days). Above this: signal stale."""

# ---------------------------------------------------------------------------
# §2 Volatility & Risk Modeling
# ---------------------------------------------------------------------------

EWMA_LAMBDA: float = 0.94
"""RiskMetrics EWMA decay factor for variance estimation."""

VAR_CONFIDENCE_95: float = 0.05
"""VaR confidence level α = 0.05 (95% VaR)."""

VAR_CONFIDENCE_99: float = 0.01
"""VaR confidence level α = 0.01 (99% VaR)."""

ZSCORE_ENTRY_LOW: float = 2.0
"""Minimum z-score threshold for signal entry."""

ZSCORE_ENTRY_HIGH: float = 3.0
"""Maximum z-score threshold for signal entry (conservative sizing)."""

# ---------------------------------------------------------------------------
# §3 Signal Processing & Microstructure
# ---------------------------------------------------------------------------

MARKET_IMPACT_GAMMA: float = 0.5
"""Square-root law market impact exponent γ = 0.5."""

CROSS_ASSET_CORR_BASELINE_LOW: float = 0.05
"""Lower bound of cross-asset correlation baseline ρ ∈ [0.05, 0.15]."""

CROSS_ASSET_CORR_BASELINE_HIGH: float = 0.15
"""Upper bound of cross-asset correlation baseline ρ ∈ [0.05, 0.15]."""

# ---------------------------------------------------------------------------
# §4 Backtesting & Portfolio Construction
# ---------------------------------------------------------------------------

SHARPE_FLOOR_SYSTEMATIC_MACRO: float = 2.0
"""Pre-cost Sharpe floor for systematic macro (daily signals)."""

SHARPE_FLOOR_INTRADAY: float = 2.5
"""Pre-cost Sharpe floor for intraday stat-arb strategies."""

SHARPE_FLOOR_CARRY: float = 1.5
"""Pre-cost Sharpe floor for cross-asset carry strategies."""

SHARPE_FLOOR_TREND: float = 0.7
"""Pre-cost Sharpe floor for trend-following CTA strategies."""

TSTAT_SIGNIFICANCE: float = 3.0
"""Minimum t-statistic for strategy significance (t ≥ 3.0)."""

ANNUALIZATION_FACTOR: int = 252
"""Trading days per year for Sharpe/vol annualization."""

POSITION_LIMIT_MAX: float = 0.05
"""Maximum single-position weight (5% concentration limit)."""

# ---------------------------------------------------------------------------
# §5 Extended Constants
# ---------------------------------------------------------------------------

STUDENT_T_DOF_LOW: int = 4
"""Minimum Student-t degrees of freedom for fat-tail modelling."""

STUDENT_T_DOF_HIGH: int = 6
"""Maximum Student-t degrees of freedom for fat-tail modelling."""

CALMAR_FLOOR: float = 0.5
"""Minimum acceptable Calmar ratio (Ann. Return / Max Drawdown)."""

ADV_PARTICIPATION_CAP: float = 0.10
"""Maximum fraction of daily ADV to trade (capacity constraint)."""

COVARIANCE_SHRINKAGE_LOW: float = 0.1
"""Ledoit-Wolf shrinkage intensity lower bound α_LW ∈ [0.1, 0.3]."""

COVARIANCE_SHRINKAGE_HIGH: float = 0.3
"""Ledoit-Wolf shrinkage intensity upper bound."""

# §5.8 Volatility Regime Thresholds
VIX_RISK_ON_THRESHOLD: float = 20.0
"""VIX level below which market is in risk-on regime (full signal sizing)."""

VIX_CRISIS_THRESHOLD: float = 30.0
"""VIX level above which market is in crisis regime (minimal sizing)."""

VIX_RISK_ON_SCALE: float = 1.00
"""Signal scale factor in risk-on regime (VIX < 20)."""

VIX_TRANSITION_SCALE: float = 0.60
"""Signal scale factor in transition regime (20 ≤ VIX < 30)."""

VIX_CRISIS_SCALE: float = 0.25
"""Signal scale factor in crisis regime (VIX ≥ 30)."""

# ---------------------------------------------------------------------------
# §4.1 Sharpe Decay Waterfall
# ---------------------------------------------------------------------------

SHARPE_DECAY_TRANSACTION_COSTS: float = 0.40
"""Expected Sharpe reduction from bid-ask, commissions, financing."""

SHARPE_DECAY_OVERFITTING: float = 0.30
"""Expected Sharpe reduction from IS bias and data-snooping."""

SHARPE_DECAY_LIVE_SLIPPAGE: float = 0.15
"""Expected Sharpe reduction from market impact and signal decay."""

# ---------------------------------------------------------------------------
# Signal Health Monitoring
# ---------------------------------------------------------------------------

IC_FLOOR: float = 0.02
"""Minimum acceptable Information Coefficient (retire below this)."""

ICIR_FLOOR: float = 0.50
"""Minimum acceptable IC Information Ratio."""

ROLLING_ICIR_WINDOW: int = 60
"""Rolling window (days) for ICIR computation."""

HALF_LIFE_BREACH_THRESHOLD: float = 0.85
"""Fraction of expected half-life remaining before retirement alert."""

MAX_R2_ORTHOGONALITY: float = 0.15
"""Maximum pairwise R² allowed between signals (orthogonality gate)."""

MAX_VIF_ORTHOGONALITY: float = 5.0
"""Maximum VIF allowed per signal (multicollinearity gate)."""

# ---------------------------------------------------------------------------
# HLS Branch — ISCF & MGD Signal Constants
# ---------------------------------------------------------------------------

# ISCF — Idiosyncratic Supply Chain Flow (Metals/Energy)
ISCF_BACKWARDATION_ZSCORE_THRESHOLD: float = 1.5
"""Z-score threshold for steep backwardation signal entry."""

ISCF_VOL_NORMALISATION_WINDOW: int = 20
"""Rolling window (days) for realized vol normalisation of basis."""

ISCF_INVENTORY_DECAY_LAMBDA: float = 0.90
"""EMA decay for rolling inventory-adjusted basis."""

ISCF_MAX_BASIS_ZSCORE: float = 4.0
"""Winsorisation cap on basis z-score to prevent blow-up."""

COMMODITY_ASSETS: tuple[str, ...] = (
    "WTI_CL", "BRENT_CO", "NGAS_NG",
    "COPPER_HG", "GOLD_GC", "SILVER_SI",
    "ALUMINIUM_LA", "ZINC_LX",
)

# MGD — Real-Time Macro Growth Divergence (FX)
MGD_SURPRISE_EMA_SPAN: int = 21
"""EMA span (days) for nowcast surprise smoothing. 21d → half-life ≈ 14.5d inside [21,63]d."""

MGD_FORWARD_CURVE_WINDOW: int = 21
"""Rolling window for FX forward curve expectation estimation."""

MGD_ZSCORE_WINDOW: int = 60
"""Rolling window for MGD cross-sectional z-score."""

MGD_PMI_WEIGHT: float = 0.40
"""Weight on PMI component in composite MGD surprise index."""

MGD_INFLATION_WEIGHT: float = 0.30
"""Weight on CPI surprise component."""

MGD_EMPLOYMENT_WEIGHT: float = 0.30
"""Weight on employment surprise component."""

FX_ASSETS: tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "NOKUSD", "SEKUSD",
)

# Causal Validation Framework
GRANGER_LAG_PERIODS: int = 2
"""Number of lag periods for intraday Granger causality VARX."""

GRANGER_PVALUE_THRESHOLD: float = 0.05
"""Max p-value for Granger causality to pass (rejects if >=)."""

CMI_ALPHA_THRESHOLD: float = 0.05
"""Conditional mutual information independence test threshold."""

CAUSAL_GAMMA_HIGH: float = 0.95
"""Causal confidence γ for full causal validation pass."""

CAUSAL_GAMMA_MEDIUM: float = 0.30
"""Causal confidence γ for predictive-only (no causal) signals."""

CAUSAL_GAMMA_REJECT: float = 0.0
"""Causal confidence γ for DoWhy placebo failure."""

NEWEY_WEST_LAG_TRUNCATION: int = 12
"""HAC Newey-West bandwidth L for macro time-series refutations."""

BLOCK_BOOTSTRAP_BLOCK_SIZE: int = 20
"""Moving block bootstrap block length b (preserves autocorrelation)."""

DOWHY_BOOTSTRAP_REPS: int = 200
"""Number of bootstrap replications for DoWhy refutation p-values."""

CAUSAL_PLACEBO_PVALUE_FLOOR: float = 0.05
"""Min p-value for placebo test (signal must be > noise at this level)."""

# Six-Month Plan KPIs
WALKFORWARD_SHARPE_TARGET: float = 0.70
"""Month-6 KPI: walk-forward Sharpe > 0.7 for ISCF and MGD."""

CORRELATION_SUPPRESSION_TARGET: float = 0.15
"""Month-6 KPI: max |ρ| with legacy signals < 0.15."""

FDR_REDUCTION_TARGET: float = 0.30
"""Month-6 KPI: portfolio FDR reduced by ≥ 30%."""
