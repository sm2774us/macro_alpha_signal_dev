# citadel_alpha/rebalance.py — Intraday rebalancing engine.
# Google Python Style Guide.

"""4x-daily intraday rebalancing logic for ISCF + MGD signals.

At Proprietary Trading Firm, strategies rebalance ~4x daily across FX, commodities,
futures, and rates at a mid-to-high frequency. This module computes:
  1. Target weights from current signal scores (γ-scaled)
  2. Required trades (Δweight) given current holdings
  3. Rebalancing cost estimate (TCA: bid-ask + market impact)
  4. Go/No-Go decision per session (London/NY/Asia)

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from citadel_alpha import constants as C

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class Session(Enum):
    """Proprietary Trading Firm — 4x daily trading sessions."""

    ASIA = "Asia"
    LONDON = "London"
    NY_OPEN = "NY_Open"
    NY_CLOSE = "NY_Close"


@dataclass
class RebalanceDecision:
    """Output of the rebalancing engine for one session."""

    session: Session
    signal_name: str
    current_weights: FloatArray  # Current portfolio weights (N,)
    target_weights: FloatArray  # Signal-implied target weights (N,)
    delta_weights: FloatArray  # Required trades Δw (N,)
    turnover: float  # |Δw|.sum() / 2
    estimated_tc_bps: float  # Estimated transaction cost in bps
    causal_gamma: float  # γ from causal validation
    go_decision: bool  # True = execute rebalance
    reason: str  # Explanation string


def _hrp_weights(scores: FloatArray, n: int) -> FloatArray:
    """Simple equal-risk allocation scaled by |signal score|."""
    abs_scores = np.abs(scores)
    total = float(np.sum(abs_scores))
    if total < 1e-10:
        return np.ones(n) / n
    raw = abs_scores / total
    # Cap at POSITION_LIMIT_MAX
    raw = np.minimum(raw, C.POSITION_LIMIT_MAX * n)
    raw /= float(np.sum(raw))
    return raw * np.sign(scores)


def compute_rebalance(
    signal_name: str,
    rank_scores: FloatArray,
    current_weights: FloatArray,
    causal_gamma: float,
    session: Session,
    bid_ask_bps: float = 5.0,
    adv_fraction: float = C.ADV_PARTICIPATION_CAP,
) -> RebalanceDecision:
    """Compute rebalancing decision for one signal at one session.

    Applies causal γ scaling to target weights:
        w_target = γ · HRP_weights(rank_scores)

    Go/No-Go criteria (pre-agreed kill criteria):
        - γ < 0.25: suspend (causal validation failed)
        - turnover > 0.50: spread over 2 sessions
        - estimated TC > 50% of expected gross alpha: hold

    Args:
        signal_name: "ISCF" or "MGD".
        rank_scores: Gaussian-rank-normalised scores (N,).
        current_weights: Current portfolio weights (N,), sum ~ 1.0.
        causal_gamma: γ from causal stack (0.0, 0.30, or 0.95).
        session: Current trading session.
        bid_ask_bps: Bid-ask spread estimate in basis points.
        adv_fraction: Max fraction of ADV to trade.

    Returns:
        RebalanceDecision with go/no-go and cost estimate.
    """
    n = len(rank_scores)
    scores = np.asarray(rank_scores, dtype=np.float64)
    curr = np.asarray(current_weights, dtype=np.float64)

    # γ-scaled target weights
    raw_target = _hrp_weights(scores, n)
    target = causal_gamma * raw_target

    delta = target - curr
    turnover = float(np.sum(np.abs(delta))) / 2.0

    # TCA estimate: bid-ask + sqrt-law market impact
    # TC (bps) = bid_ask_bps/2 * turnover + γ_impact * sqrt(turnover/ADV)
    impact_bps = 10.0 * math.sqrt(max(turnover / max(adv_fraction, 1e-8), 0))
    tc_bps = bid_ask_bps / 2.0 * turnover * 1e4 + impact_bps

    # Expected gross alpha proxy (IC * vol * sqrt(N) * 252/4 sessions)
    expected_alpha_bps = max(float(np.mean(np.abs(scores))), 0.01) * 10000.0 / 4.0

    # Kill criteria
    if causal_gamma < 0.25:
        go = False
        reason = f"γ={causal_gamma:.2f} < 0.25 threshold — causal validation suspended"
    elif turnover > 0.50:
        go = False
        reason = f"Turnover={turnover:.2%} > 50% — split across 2 sessions"
    elif tc_bps > 0.5 * expected_alpha_bps:
        go = False
        reason = (
            f"TC={tc_bps:.1f}bps > 50% of expected alpha={expected_alpha_bps:.1f}bps"
        )
    else:
        go = True
        reason = (
            f"Session={session.value} γ={causal_gamma:.2f} "
            f"turnover={turnover:.2%} TC={tc_bps:.1f}bps ✓"
        )

    return RebalanceDecision(
        session=session,
        signal_name=signal_name,
        current_weights=curr,
        target_weights=target,
        delta_weights=delta,
        turnover=turnover,
        estimated_tc_bps=tc_bps,
        causal_gamma=causal_gamma,
        go_decision=go,
        reason=reason,
    )


def rebalance_summary(decisions: list[RebalanceDecision]) -> str:
    """Format rebalancing decisions as a human-readable summary."""
    lines = ["", "── Rebalancing Decisions ──────────────────────────"]
    for d in decisions:
        icon = "✓ GO" if d.go_decision else "✗ HOLD"
        lines.append(
            f"  {d.signal_name} [{d.session.value}]: {icon} | "
            f"Turnover={d.turnover:.2%} | TC≈{d.estimated_tc_bps:.1f}bps | {d.reason}"
        )
    return "\n".join(lines)
