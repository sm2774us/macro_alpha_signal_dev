#!/usr/bin/env python3
# Copyright 2025 HLS Trading
# scripts/generate_kpi_report.py
# Google Python Style Guide.
"""Generate a markdown KPI report from signal health JSON artifacts.

Called by GitHub Actions signal-health-monitor job.
Output is posted to $GITHUB_STEP_SUMMARY for inline PR/run visibility.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _status(val: float, floor: float, invert: bool = False) -> str:
    ok = val >= floor if not invert else val <= floor
    return "✅" if ok else "⚠️"


def generate_report(
    artifacts_dir: Path,
    output: Path,
    session: str,
    commit: str,
    ref: str,
) -> None:
    """Read JSON artifacts and write KPI markdown report."""

    signals = ["iscf", "mgd"]
    data: dict[str, dict] = {}
    for sig in signals:
        fp = artifacts_dir / f"{sig}_health.json"
        if fp.exists():
            data[sig] = json.loads(fp.read_text())

    lines: list[str] = [
        "# 📊 HLS Alpha Engine — Signal KPI Report",
        "",
        f"**Session:** {session}  |  **Ref:** `{ref}`  |  **Commit:** `{commit[:8]}`",
        "",
        "## Portfolio KPIs",
        "",
        "| Signal | Gross SR | Walk-fwd Floor | IC | ICIR | Half-Life (d) | Causal γ | Status |",
        "|--------|----------|---------------|-----|------|--------------|---------|--------|",
    ]

    any_retire = False
    for sig in signals:
        if sig not in data:
            lines.append(f"| {sig.upper()} | N/A | — | — | — | — | — | ❓ |")
            continue
        d = data[sig]
        sr = d.get("gross_sr", 0.0)
        ic = d.get("mean_ic", 0.0)
        icir = d.get("icir", 0.0)
        hl = d.get("half_life_days", float("inf"))
        gamma = d.get("causal_gamma", 0.0)
        retire = d.get("retirement_recommended", True)
        passes_sr = d.get("passes_sr_floor", False)
        if retire:
            any_retire = True
        hl_str = f"{hl:.1f}" if math.isfinite(hl) else "∞"
        status = "✅ LIVE" if not retire else "⚠️ REVIEW"
        lines.append(
            f"| {sig.upper()} | {sr:.3f} | {_status(sr, 0.70)} | "
            f"{ic:.4f} | {icir:.4f} | {hl_str} | {gamma:.2f} | {status} |"
        )

    lines += [
        "",
        "## Signal Retirement Analysis (Half-Life Gate)",
        "",
        "| Signal | Half-Life (d) | Min (d) | Max (d) | Status |",
        "|--------|--------------|---------|---------|--------|",
    ]
    for sig in signals:
        if sig not in data:
            continue
        d = data[sig]
        hl = d.get("half_life_days", float("inf"))
        hl_str = f"{hl:.1f}" if math.isfinite(hl) else "∞"
        alive = 21.0 <= hl <= 63.0 if math.isfinite(hl) else False
        alert = hl < 21.0 * 0.85 if math.isfinite(hl) else True
        status = "✅ Alive" if alive else ("🚨 RETIRE" if alert else "⚠️ Decaying")
        lines.append(f"| {sig.upper()} | {hl_str} | 21 | 63 | {status} |")

    lines += [
        "",
        "## Causal Validation Summary",
        "",
        "| Signal | Granger p | Placebo p | Policy p | Recommendation | γ |",
        "|--------|-----------|-----------|----------|---------------|---|",
    ]
    for sig in signals:
        if sig not in data:
            continue
        d = data[sig]
        gp = d.get("granger_p", 1.0)
        pp = d.get("placebo_p", 0.0)
        polp = d.get("policy_invariance_p", 0.0)
        rec = d.get("causal_recommendation", "UNKNOWN")
        gamma = d.get("causal_gamma", 0.0)
        gok = "✅" if gp < 0.05 else "⚠️"
        pok = "✅" if pp > 0.05 else "⚠️"
        polk = "✅" if polp > 0.05 else "⚠️"
        lines.append(
            f"| {sig.upper()} | {gok} {gp:.4f} | {pok} {pp:.4f} | {polk} {polp:.4f} | {rec} | {gamma:.2f} |"
        )

    lines += [
        "",
        "## Rebalancing Recommendation",
        "",
    ]
    for sig in signals:
        if sig not in data:
            continue
        d = data[sig]
        gamma = d.get("causal_gamma", 0.0)
        sr = d.get("gross_sr", 0.0)
        retire = d.get("retirement_recommended", True)
        if retire:
            lines.append(f"- **{sig.upper()}**: 🚨 `RETIRE` — Freeze allocation. Begin new hypothesis search.")
        elif gamma >= 0.90:
            lines.append(f"- **{sig.upper()}**: ✅ `FULL ALLOCATION` — γ={gamma:.2f}. Maintain 4x daily rebalance.")
        elif gamma >= 0.25:
            lines.append(f"- **{sig.upper()}**: ⚠️ `REDUCED ALLOCATION` — γ={gamma:.2f}. Cap at 50% target weight.")
        else:
            lines.append(f"- **{sig.upper()}**: 🚨 `SUSPEND` — γ={gamma:.2f}. Causal validation failed.")

    lines += [
        "",
        "## Sharpe Waterfall (Pre-Cost → Net)",
        "",
        "| Signal | Gross SR | After TC (−0.40) | After Overfit (−0.30) | After Slippage (−0.15) | Net SR |",
        "|--------|----------|-----------------|----------------------|----------------------|--------|",
    ]
    for sig in signals:
        if sig not in data:
            continue
        sr = data[sig].get("gross_sr", 0.0)
        sr_tc = sr - 0.40
        sr_ov = sr_tc - 0.30
        sr_sl = sr_ov - 0.15
        lines.append(
            f"| {sig.upper()} | {sr:.3f} | {sr_tc:.3f} | {sr_ov:.3f} | {sr_sl:.3f} | {sr_sl:.3f} |"
        )

    if any_retire:
        lines += ["", "---", "**🚨 ACTION REQUIRED: One or more signals recommend retirement. "
                  "Review half-life and IC trends. Begin next hypothesis per 5-step falsification protocol.**"]
    else:
        lines += ["", "---", "**✅ All signals healthy. Continue 4x daily rebalancing.**"]

    output.write_text("\n".join(lines))
    print(f"KPI report written: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/KPI_REPORT.md"))
    parser.add_argument("--session", default="manual")
    parser.add_argument("--commit", default="local")
    parser.add_argument("--ref", default="local")
    args = parser.parse_args()
    generate_report(args.artifacts_dir, args.output, args.session, args.commit, args.ref)


if __name__ == "__main__":
    main()
