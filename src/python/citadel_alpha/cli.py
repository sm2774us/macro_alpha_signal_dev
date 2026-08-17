# citadel_alpha/cli.py — Proprietary Trading Firm — Alpha Engine CLI.
# Google Python Style Guide.

"""Click CLI for Proprietary Trading Firm — Alpha Engine (ISCF + MGD).

Commands:
    hls-run       — full ISCF+MGD pipeline on synthetic data
    hls-live      — fetch real market data via yfinance and run signals
    hls-monitor   — signal KPI health check (for GitHub Actions cron)
    benchmark     — C++ vs Python timing comparison
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import click
import numpy as np

logging.basicConfig(level=logging.WARNING)


@click.group()
@click.version_option("2.0.0", prog_name="hls-alpha")
def cli() -> None:
    """Proprietary Trading Firm — Alpha Engine — ISCF & MGD Systematic Macro Signals."""


# ---------------------------------------------------------------------------
# hls-run: synthetic data pipeline
# ---------------------------------------------------------------------------


@cli.command("hls-run")
@click.option("--n-assets", default=8, show_default=True)
@click.option("--n-periods", default=2000, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output-dir", default="artifacts", show_default=True)
def hls_run(n_assets: int, n_periods: int, seed: int, output_dir: str) -> None:
    """Run ISCF + MGD on synthetic panel with causal validation."""
    from citadel_alpha import (
        data_hls,
        signals_hls,
        causal,
        falsification as fal,
        constants as C,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(click.style("━" * 60, fg="cyan"))
    click.echo(
        click.style(
            "  Proprietary Trading Firm — Alpha Engine — ISCF + MGD (Synthetic)",
            fg="cyan",
            bold=True,
        )
    )
    click.echo(click.style("━" * 60, fg="cyan"))

    warmup = 120

    for signal_name, gen_fn, compute_fn, panel_attr in [
        (
            "ISCF",
            data_hls.generate_commodity_panel,
            lambda p, i: signals_hls.compute_iscf(
                p.spot[i],
                p.deferred[i],
                p.rvol[i],
                p.forward_returns[i],
                p.macro_beta[i],
                np.column_stack(
                    [p.trend_returns[i], p.momentum_returns[i], p.carry_returns[i]]
                ),
            ),
            None,
        ),
        (
            "MGD",
            data_hls.generate_fx_panel,
            lambda p, i: signals_hls.compute_mgd(
                p.pmi_surprise[i],
                p.cpi_surprise[i],
                p.emp_surprise[i],
                p.fwd_expectation[i],
                p.roll_std[i],
                p.forward_returns[i],
                np.column_stack(
                    [p.trend_returns[i], p.momentum_returns[i], p.carry_returns[i]]
                ),
            ),
            None,
        ),
    ]:
        click.echo(click.style(f"\n[{signal_name}] Computing...", fg="yellow"))
        panel = gen_fn(n=n_assets, t=n_periods, seed=seed)
        ics, pnl_list = [], []

        for i in range(warmup, n_periods):
            res = compute_fn(panel, i)
            ics.append(res.ic)
            pnl_list.append(float(np.mean(res.rank_score * panel.forward_returns[i])))

        ic_arr = np.array(ics)
        pnl_arr = np.array(pnl_list)
        sr = (
            float(np.mean(pnl_arr))
            / max(float(np.std(pnl_arr, ddof=1)), 1e-8)
            * math.sqrt(252)
        )

        # Causal stack on first 500-warmup periods
        n_causal = min(500, n_periods - warmup)
        sig_slice = np.array(
            [
                float(np.mean(compute_fn(panel, i).rank_score))
                for i in range(warmup, warmup + n_causal)
            ]
        )
        ret_slice = np.array(
            [
                float(np.mean(panel.forward_returns[i]))
                for i in range(warmup, warmup + n_causal)
            ]
        )
        confounders = np.column_stack(
            [
                panel.session_dummies[warmup : warmup + n_causal],
                panel.vix_proxy[warmup : warmup + n_causal].reshape(-1, 1),
            ]
        )
        causal_result = causal.run_causal_stack(
            signal_name, sig_slice, ret_slice, confounders=confounders, n_bootstrap=100
        )
        health = fal.signal_health_report(signal_name, ic_arr, sr, n_obs=len(pnl_arr))

        color = "green" if causal_result.recommendation == "PASS" else "yellow"
        click.echo(click.style(f"\n{causal_result.summary}", fg=color))
        click.echo(
            click.style(
                f"\n{health.summary}",
                fg="green" if not health.retirement_recommended else "red",
            )
        )

        artifact = {
            "signal": signal_name,
            "gross_sr": sr,
            "mean_ic": health.mean_ic,
            "icir": health.icir,
            "half_life_days": health.half_life.half_life_days,
            "retirement_recommended": health.retirement_recommended,
            "causal_gamma": causal_result.final_gamma,
            "causal_recommendation": causal_result.recommendation,
            "granger_p": causal_result.granger.p_value,
            "placebo_p": causal_result.dowhy.placebo_p_value,
            "passes_sr_floor": sr >= C.WALKFORWARD_SHARPE_TARGET,
        }
        (out / f"{signal_name.lower()}_health.json").write_text(
            json.dumps(artifact, indent=2)
        )

    click.echo(click.style(f"\n✓ Artifacts written to {out}/", fg="cyan", bold=True))


# ---------------------------------------------------------------------------
# hls-live: real market data via yfinance
# ---------------------------------------------------------------------------


@cli.command("hls-live")
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default="2024-12-31", show_default=True)
@click.option(
    "--provider",
    default="yfinance",
    show_default=True,
    type=click.Choice(["yfinance", "proprietary"]),
)
@click.option("--output-dir", default="artifacts", show_default=True)
def hls_live(start: str, end: str, provider: str, output_dir: str) -> None:
    """Fetch real market data and run ISCF+MGD signals end-to-end."""
    from citadel_alpha import signals_hls, causal, falsification as fal, constants as C
    from citadel_alpha.data_provider import get_iscf_provider, get_mgd_provider

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(click.style("━" * 60, fg="cyan"))
    click.echo(
        click.style(
            f"  Proprietary Trading Firm — Live Run — Provider: {provider}",
            fg="cyan",
            bold=True,
        )
    )
    click.echo(click.style("━" * 60, fg="cyan"))

    # ── ISCF ──
    click.echo(click.style("\n[ISCF] Fetching commodity data...", fg="yellow"))
    iscf_provider = get_iscf_provider(mode=provider)
    try:
        iscf_data = iscf_provider.fetch(start=start, end=end)
        t, n = iscf_data.spot.shape
        warmup = min(252, t // 4)

        ics, pnl_list = [], []
        for i in range(warmup, t):
            baseline = np.column_stack(
                [
                    iscf_data.trend_baseline[i],
                    iscf_data.momentum_baseline[i],
                    iscf_data.carry_baseline[i],
                ]
            )
            res = signals_hls.compute_iscf(
                iscf_data.spot[i],
                iscf_data.deferred[i],
                iscf_data.rvol[i],
                iscf_data.forward_returns[i],
                iscf_data.macro_beta[i],
                baseline,
            )
            ics.append(res.ic)
            pnl_list.append(
                float(np.mean(res.rank_score * iscf_data.forward_returns[i]))
            )

        ic_arr, pnl_arr = np.array(ics), np.array(pnl_list)
        sr = (
            float(np.mean(pnl_arr))
            / max(float(np.std(pnl_arr, ddof=1)), 1e-8)
            * math.sqrt(252)
        )
        health = fal.signal_health_report("ISCF", ic_arr, sr, n_obs=len(pnl_arr))
        click.echo(
            click.style(
                f"\n{health.summary}",
                fg="green" if not health.retirement_recommended else "red",
            )
        )
        (out / "iscf_live_health.json").write_text(
            json.dumps(
                {
                    "signal": "ISCF",
                    "gross_sr": sr,
                    "mean_ic": health.mean_ic,
                    "half_life_days": health.half_life.half_life_days,
                    "retirement_recommended": health.retirement_recommended,
                    "n_assets": n,
                    "n_obs": t,
                    "start": start,
                    "end": end,
                },
                indent=2,
            )
        )
    except Exception as e:
        click.echo(click.style(f"[ISCF] Error: {e}", fg="red"))

    # ── MGD ──
    click.echo(click.style("\n[MGD] Fetching FX+macro data...", fg="yellow"))
    mgd_provider = get_mgd_provider(mode=provider)
    try:
        mgd_data = mgd_provider.fetch(start=start, end=end)
        t, n = mgd_data.forward_returns.shape
        warmup = min(252, t // 4)

        ics, pnl_list = [], []
        for i in range(warmup, t):
            baseline = np.column_stack(
                [
                    mgd_data.trend_baseline[i],
                    mgd_data.momentum_baseline[i],
                    mgd_data.carry_baseline[i],
                ]
            )
            res = signals_hls.compute_mgd(
                mgd_data.pmi_surprise[i],
                mgd_data.cpi_surprise[i],
                mgd_data.emp_surprise[i],
                mgd_data.fwd_expectation[i],
                mgd_data.roll_std[i],
                mgd_data.forward_returns[i],
                baseline,
            )
            ics.append(res.ic)
            pnl_list.append(
                float(np.mean(res.rank_score * mgd_data.forward_returns[i]))
            )

        ic_arr, pnl_arr = np.array(ics), np.array(pnl_list)
        sr = (
            float(np.mean(pnl_arr))
            / max(float(np.std(pnl_arr, ddof=1)), 1e-8)
            * math.sqrt(252)
        )
        health = fal.signal_health_report("MGD", ic_arr, sr, n_obs=len(pnl_arr))
        click.echo(
            click.style(
                f"\n{health.summary}",
                fg="green" if not health.retirement_recommended else "red",
            )
        )
        (out / "mgd_live_health.json").write_text(
            json.dumps(
                {
                    "signal": "MGD",
                    "gross_sr": sr,
                    "mean_ic": health.mean_ic,
                    "half_life_days": health.half_life.half_life_days,
                    "retirement_recommended": health.retirement_recommended,
                    "n_assets": n,
                    "n_obs": t,
                    "start": start,
                    "end": end,
                },
                indent=2,
            )
        )
    except Exception as e:
        click.echo(click.style(f"[MGD] Error: {e}", fg="red"))

    click.echo(
        click.style(f"\n✓ Live artifacts written to {out}/", fg="cyan", bold=True)
    )


# ---------------------------------------------------------------------------
# hls-monitor: cron health check
# ---------------------------------------------------------------------------


@cli.command("hls-monitor")
@click.option("--n-assets", default=8, show_default=True)
@click.option("--n-periods", default=2000, show_default=True)
@click.option("--output-dir", default="artifacts", show_default=True)
@click.option("--sr-floor", default=0.70, show_default=True)
def hls_monitor(
    n_assets: int, n_periods: int, output_dir: str, sr_floor: float
) -> None:
    """Signal KPI health check + half-life gate + rebalancing decisions (4x daily cron)."""
    from citadel_alpha import (
        data_hls,
        signals_hls,
        causal,
        falsification as fal,
        constants as C,
    )
    from citadel_alpha.rebalance import compute_rebalance, rebalance_summary, Session
    import math

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(click.style("━" * 70, fg="cyan"))
    click.echo(
        click.style(
            "  Proprietary Trading Firm — Signal Health Monitor + Rebalancing Engine",
            fg="cyan",
            bold=True,
        )
    )
    click.echo(click.style("━" * 70, fg="cyan"))

    warmup = 120
    panels = {
        "ISCF": data_hls.generate_commodity_panel(n=n_assets, t=n_periods, seed=42),
        "MGD": data_hls.generate_fx_panel(n=n_assets, t=n_periods, seed=42),
    }

    any_issue = False
    rebalance_decisions = []

    for sig_name, panel in panels.items():
        ics, pnl_list, last_ranks = [], [], None

        for i in range(warmup, n_periods):
            if sig_name == "ISCF":
                bl = np.column_stack(
                    [
                        panel.trend_returns[i],
                        panel.momentum_returns[i],
                        panel.carry_returns[i],
                    ]
                )
                res = signals_hls.compute_iscf(
                    panel.spot[i],
                    panel.deferred[i],
                    panel.rvol[i],
                    panel.forward_returns[i],
                    panel.macro_beta[i],
                    bl,
                )
            else:
                bl = np.column_stack(
                    [
                        panel.trend_returns[i],
                        panel.momentum_returns[i],
                        panel.carry_returns[i],
                    ]
                )
                res = signals_hls.compute_mgd(
                    panel.pmi_surprise[i],
                    panel.cpi_surprise[i],
                    panel.emp_surprise[i],
                    panel.fwd_expectation[i],
                    panel.roll_std[i],
                    panel.forward_returns[i],
                    bl,
                )
            ics.append(res.ic)
            pnl_list.append(float(np.mean(res.rank_score * panel.forward_returns[i])))
            last_ranks = res.rank_score

        ic_arr = np.array(ics)
        pnl_arr = np.array(pnl_list)
        sr = (
            float(np.mean(pnl_arr))
            / max(float(np.std(pnl_arr, ddof=1)), 1e-8)
            * math.sqrt(252)
        )

        # Causal stack on rolling window
        n_c = min(500, n_periods - warmup)
        sig_slice = np.array(
            [
                float(
                    np.mean(
                        (
                            signals_hls.compute_iscf
                            if sig_name == "ISCF"
                            else signals_hls.compute_mgd
                        )(
                            *(
                                [
                                    panel.spot[i],
                                    panel.deferred[i],
                                    panel.rvol[i],
                                    panel.forward_returns[i],
                                    panel.macro_beta[i],
                                ]
                                if sig_name == "ISCF"
                                else [
                                    panel.pmi_surprise[i],
                                    panel.cpi_surprise[i],
                                    panel.emp_surprise[i],
                                    panel.fwd_expectation[i],
                                    panel.roll_std[i],
                                    panel.forward_returns[i],
                                ]
                            ),
                            np.column_stack(
                                [
                                    panel.trend_returns[i],
                                    panel.momentum_returns[i],
                                    panel.carry_returns[i],
                                ]
                            ),
                        ).rank_score
                    )
                )
                for i in range(warmup, warmup + n_c)
            ]
        )
        ret_slice = np.array(
            [
                float(np.mean(panel.forward_returns[i]))
                for i in range(warmup, warmup + n_c)
            ]
        )
        causal_result = causal.run_causal_stack(
            sig_name, sig_slice, ret_slice, n_bootstrap=50
        )

        health = fal.signal_health_report(sig_name, ic_arr, sr, n_obs=len(pnl_arr))
        color = "green" if not health.retirement_recommended else "red"
        click.echo(click.style(f"\n{health.summary}", fg=color))
        click.echo(
            click.style(
                f"\n{causal_result.summary}",
                fg="green" if causal_result.recommendation == "PASS" else "yellow",
            )
        )

        if health.retirement_recommended or sr < sr_floor:
            any_issue = True

        # Rebalancing decision for each session
        current_w = np.ones(n_assets) / n_assets  # Placeholder equal-weight
        for session in Session:
            dec = compute_rebalance(
                sig_name, last_ranks, current_w, causal_result.final_gamma, session
            )
            rebalance_decisions.append(dec)

        artifact = {
            "signal": sig_name,
            "gross_sr": sr,
            "mean_ic": health.mean_ic,
            "icir": health.icir,
            "half_life_days": health.half_life.half_life_days,
            "half_life_is_alive": health.half_life.is_alive,
            "retirement_recommended": health.retirement_recommended,
            "causal_gamma": causal_result.final_gamma,
            "causal_recommendation": causal_result.recommendation,
            "granger_p": causal_result.granger.p_value,
            "placebo_p": causal_result.dowhy.placebo_p_value,
            "policy_invariance_p": causal_result.dowhy.policy_invariance_p_value,
            "passes_sr_floor": sr >= sr_floor,
        }
        (out / f"{sig_name.lower()}_health.json").write_text(
            json.dumps(artifact, indent=2)
        )

    click.echo(click.style(rebalance_summary(rebalance_decisions), fg="cyan"))

    click.echo()
    if any_issue:
        click.echo(
            click.style(
                "\n⚠ One or more signals below KPI floor or recommend retirement.",
                fg="yellow",
                bold=True,
            )
        )
        sys.exit(1)
    else:
        click.echo(
            click.style(
                "\n✓ All signals healthy. Continue 4x daily rebalancing.",
                fg="green",
                bold=True,
            )
        )


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@cli.command("benchmark")
@click.option("--n-assets", default=8, show_default=True)
@click.option("--n-reps", default=5000, show_default=True)
def benchmark(n_assets: int, n_reps: int) -> None:
    """Benchmark C++ hot-path vs pure-Python for ISCF/MGD."""
    import time
    from citadel_alpha import data_hls, signals_hls

    panel = data_hls.generate_commodity_panel(n=n_assets, t=300, seed=0)
    i = 150
    baseline = np.column_stack(
        [panel.trend_returns[i], panel.momentum_returns[i], panel.carry_returns[i]]
    )

    click.echo(click.style("━" * 50, fg="cyan"))
    click.echo(
        click.style(
            "  Proprietary Trading Firm — Benchmark — ISCF Python hot-path",
            fg="cyan",
            bold=True,
        )
    )
    start = time.perf_counter()
    for _ in range(n_reps):
        signals_hls.compute_iscf(
            panel.spot[i],
            panel.deferred[i],
            panel.rvol[i],
            panel.forward_returns[i],
            panel.macro_beta[i],
            baseline,
        )
    elapsed = time.perf_counter() - start
    click.echo(
        f"  ISCF: {n_reps} reps in {elapsed:.3f}s "
        f"({elapsed / n_reps * 1e6:.1f} μs/rep)"
    )
    click.echo(click.style("━" * 50, fg="cyan"))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
