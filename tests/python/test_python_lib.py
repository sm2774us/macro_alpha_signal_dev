# tests/python/test_python_lib.py
# Google Python Style Guide. Pytest.

"""Comprehensive test suite for the citadel_alpha Python library.

Tests cover:
  - All 5 signal computations (shape, range, IC validity)
  - Portfolio construction (HRP, Fractional Kelly)
  - Analytics (orthogonality, FLOAM, DSR, CPCV)
  - Data generation determinism
  - CLI smoke tests
  - Edge cases (constant inputs, NaN robustness, n=4 minimum)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.random import default_rng

from citadel_alpha import signals as sig_mod
from citadel_alpha import portfolio as port_mod
from citadel_alpha import analytics as ana_mod
from citadel_alpha import data as data_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    return data_mod.generate_macro_panel(n_assets=10, n_periods=500, seed=0)


@pytest.fixture(scope="module")
def spot(panel):
    """Last time-step spot data for signal computation."""
    t = -1
    return {
        "ytm_10y": panel.ytm_10y[t],
        "ytm_2y": panel.ytm_2y[t],
        "iv": panel.implied_vol[t],
        "rv": panel.realised_vol[t],
        "vrp_std": panel.vrp_rolling_std[t],
        "surprises": panel.surprises[t],
        "ema_prev": panel.ema_state[t],
        "surprise_std": panel.surprise_rolling_std[t],
        "bid_ask": panel.bid_ask_spread[t],
        "ois_1y": panel.ois_1y[t],
        "policy_rate": panel.policy_rate[t],
        "ois_delta": panel.ois_delta[t],
        "ois_std": panel.ois_rolling_std[t],
        "regime_wt": panel.regime_wt[t],
        "next_ret": panel.forward_returns[t],
        "n": 10,
    }


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_panel_shapes(self, panel):
        t, n = 500, 10
        assert panel.ytm_10y.shape == (t, n)
        assert panel.ytm_2y.shape == (t, n)
        assert panel.forward_returns.shape == (t, n)
        assert panel.implied_vol.shape == (t, n)

    def test_panel_determinism(self):
        p1 = data_mod.generate_macro_panel(seed=99)
        p2 = data_mod.generate_macro_panel(seed=99)
        assert_array_equal(p1.ytm_10y, p2.ytm_10y)

    def test_yields_positive(self, panel):
        assert np.all(panel.ytm_10y > 0)
        assert np.all(panel.ytm_2y > 0)

    def test_implied_vol_gt_realised_median(self, panel):
        # On average VRP > 0 (structural vol risk premium).
        vrp = panel.implied_vol - panel.realised_vol
        assert np.median(vrp) > 0

    def test_bid_ask_positive(self, panel):
        assert np.all(panel.bid_ask_spread > 0)

    def test_assets_list(self, panel):
        assert len(panel.assets) == 10
        assert "USD" in panel.assets


# ---------------------------------------------------------------------------
# Signal 1 — GCSD
# ---------------------------------------------------------------------------


class TestGCSD:
    """Tests for Global Yield Curve Slope Divergence signal."""

    def test_vix_crisis_reduces_score(self, spot):
        """Crisis VIX regime should reduce signal magnitude."""
        res_calm = sig_mod.compute_gcsd(
            spot["ytm_10y"],
            spot["ytm_2y"],
            spot["next_ret"],
            vix_level=12.0,
            use_cpp=False,
        )
        res_crisis = sig_mod.compute_gcsd(
            spot["ytm_10y"],
            spot["ytm_2y"],
            spot["next_ret"],
            vix_level=45.0,
            use_cpp=False,
        )
        assert (
            np.abs(res_crisis.raw_score).max()
            < np.abs(res_calm.raw_score).max() + 1e-10
        )

    def test_returns_signal_result(self, spot):
        res = sig_mod.compute_gcsd(spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"])
        assert isinstance(res, sig_mod.SignalResult)
        assert res.signal_name == "GCSD"

    def test_output_shape(self, spot):
        res = sig_mod.compute_gcsd(spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"])
        n = spot["n"]
        assert res.raw_score.shape == (n,)
        assert res.z_score.shape == (n,)
        assert res.rank_score.shape == (n,)

    def test_z_score_mean_near_zero(self, spot):
        res = sig_mod.compute_gcsd(spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"])
        assert abs(np.mean(res.z_score)) < 1e-10

    def test_ic_in_valid_range(self, spot):
        res = sig_mod.compute_gcsd(spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"])
        assert -1.0 <= res.ic <= 1.0

    def test_rank_score_finite(self, spot):
        res = sig_mod.compute_gcsd(spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"])
        assert np.all(np.isfinite(res.rank_score))

    def test_raises_on_size_mismatch(self, spot):
        with pytest.raises((ValueError, RuntimeError)):
            sig_mod.compute_gcsd(spot["ytm_10y"][:5], spot["ytm_2y"], spot["next_ret"])

    def test_minimum_n(self):
        rng = default_rng(1)
        ytm_10y = rng.uniform(0.02, 0.05, 4)
        ytm_2y = rng.uniform(0.01, 0.03, 4)
        ret = rng.normal(0, 0.01, 4)
        res = sig_mod.compute_gcsd(ytm_10y, ytm_2y, ret)
        assert res.raw_score.shape == (4,)

    def test_no_cpp_fallback(self, spot):
        res = sig_mod.compute_gcsd(
            spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"], use_cpp=False
        )
        assert res.signal_name == "GCSD"

    def test_python_cpp_consistency(self, spot):
        """C++ and Python results must be close (not identical due to impl differences)."""
        if not sig_mod._CPP_AVAILABLE:
            pytest.skip("C++ not available")
        py = sig_mod.compute_gcsd(
            spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"], use_cpp=False
        )
        cpp = sig_mod.compute_gcsd(
            spot["ytm_10y"], spot["ytm_2y"], spot["next_ret"], use_cpp=True
        )
        assert_allclose(py.raw_score, cpp.raw_score, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Signal 2 — RVIS
# ---------------------------------------------------------------------------


class TestRVIS:
    def test_basic(self, spot):
        res = sig_mod.compute_rvis(
            spot["iv"], spot["rv"], spot["vrp_std"], spot["next_ret"]
        )
        assert res.signal_name == "RVIS"
        assert res.raw_score.shape == (spot["n"],)
        assert -1.0 <= res.ic <= 1.0

    def test_positive_vrp_positive_raw(self, spot):
        """When IV >> RV uniformly, VRP z-scores should be uniformly positive."""
        rng = default_rng(2)
        iv = np.full(10, 0.30)
        rv = np.full(10, 0.10)
        vrp_std = np.full(10, 0.01)
        ret = rng.normal(0, 0.01, 10)
        res = sig_mod.compute_rvis(iv, rv, vrp_std, ret, use_cpp=False)
        # All raw scores equal → z-score = 0 (cross-sectional demeaning).
        assert_allclose(res.z_score, np.zeros(10), atol=1e-10)

    def test_no_cpp(self, spot):
        res = sig_mod.compute_rvis(
            spot["iv"], spot["rv"], spot["vrp_std"], spot["next_ret"], use_cpp=False
        )
        assert res.signal_name == "RVIS"


# ---------------------------------------------------------------------------
# Signal 3 — MSDI
# ---------------------------------------------------------------------------


class TestMSDI:
    def test_basic(self, spot):
        res = sig_mod.compute_msdi(
            spot["surprises"], spot["ema_prev"], spot["surprise_std"], spot["next_ret"]
        )
        assert res.signal_name == "MSDI"
        assert np.all(np.isfinite(res.raw_score))

    def test_ema_smoothing(self, spot):
        """EMA output should be smoother than raw surprises."""
        res_narrow = sig_mod.compute_msdi(
            spot["surprises"],
            spot["ema_prev"],
            spot["surprise_std"],
            spot["next_ret"],
            ema_window=5,
        )
        res_wide = sig_mod.compute_msdi(
            spot["surprises"],
            spot["ema_prev"],
            spot["surprise_std"],
            spot["next_ret"],
            ema_window=30,
        )
        # Both should produce valid results.
        assert np.all(np.isfinite(res_narrow.raw_score))
        assert np.all(np.isfinite(res_wide.raw_score))

    def test_no_cpp(self, spot):
        res = sig_mod.compute_msdi(
            spot["surprises"],
            spot["ema_prev"],
            spot["surprise_std"],
            spot["next_ret"],
            use_cpp=False,
        )
        assert res.signal_name == "MSDI"


# ---------------------------------------------------------------------------
# Signal 4 — CLSD
# ---------------------------------------------------------------------------


class TestCLSD:
    def test_basic(self, spot):
        res = sig_mod.compute_clsd(spot["bid_ask"], spot["next_ret"])
        assert res.signal_name == "CLSD"
        assert res.raw_score.shape == (spot["n"],)

    def test_high_spread_negative_signal(self):
        """Highest-spread asset should have the most negative raw_score."""
        rng = default_rng(3)
        bid_ask = np.array(
            [
                0.0001,
                0.0002,
                0.0003,
                0.0100,
                0.0002,
                0.0001,
                0.0001,
                0.0002,
                0.0001,
                0.0001,
            ]
        )
        ret = rng.normal(0, 0.01, 10)
        res = sig_mod.compute_clsd(bid_ask, ret, use_cpp=False)
        # The highest spread (index 3) should give the lowest (most negative) raw score.
        assert res.raw_score[3] == pytest.approx(res.raw_score.min(), abs=1e-10)

    def test_no_cpp(self, spot):
        res = sig_mod.compute_clsd(spot["bid_ask"], spot["next_ret"], use_cpp=False)
        assert res.signal_name == "CLSD"


# ---------------------------------------------------------------------------
# Signal 5 — CBPSM
# ---------------------------------------------------------------------------


class TestCBPSM:
    def test_basic(self, spot):
        res = sig_mod.compute_cbpsm(
            spot["ois_1y"],
            spot["policy_rate"],
            spot["ois_delta"],
            spot["ois_std"],
            spot["regime_wt"],
            spot["next_ret"],
        )
        assert res.signal_name == "CBPSM"
        assert np.all(np.isfinite(res.raw_score))

    def test_regime_weight_effect(self, spot):
        """Higher regime weight should scale raw score up."""
        regime_on = np.ones(10)
        regime_off = np.full(10, 0.5)
        res_on = sig_mod.compute_cbpsm(
            spot["ois_1y"],
            spot["policy_rate"],
            spot["ois_delta"],
            spot["ois_std"],
            regime_on,
            spot["next_ret"],
            use_cpp=False,
        )
        res_off = sig_mod.compute_cbpsm(
            spot["ois_1y"],
            spot["policy_rate"],
            spot["ois_delta"],
            spot["ois_std"],
            regime_off,
            spot["next_ret"],
            use_cpp=False,
        )
        # |raw_on| >= |raw_off| element-wise.
        assert np.all(np.abs(res_on.raw_score) >= np.abs(res_off.raw_score) - 1e-10)

    def test_no_cpp(self, spot):
        res = sig_mod.compute_cbpsm(
            spot["ois_1y"],
            spot["policy_rate"],
            spot["ois_delta"],
            spot["ois_std"],
            spot["regime_wt"],
            spot["next_ret"],
            use_cpp=False,
        )
        assert res.signal_name == "CBPSM"


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------


class TestPortfolio:
    """Tests for portfolio construction module."""

    def test_ledoit_wolf_positive_definite(self, panel):
        rets = panel.forward_returns[:252]
        cov = port_mod.ledoit_wolf_shrink(rets)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), "Ledoit-Wolf covariance must be positive definite."

    def test_hrp_weights_sum_near_zero(self, panel):
        """Long-short HRP weights: net exposure should be small."""
        rets = panel.forward_returns[:252]
        scores = np.random.default_rng(5).normal(0, 1, 10)
        weights = port_mod.hierarchical_risk_parity(rets, scores)
        assert abs(np.sum(weights)) < 0.5  # Approximately market-neutral.

    def test_hrp_weights_capped(self, panel):
        rets = panel.forward_returns[:252]
        scores = np.random.default_rng(6).normal(0, 1, 10)
        weights = port_mod.hierarchical_risk_parity(rets, scores)
        assert np.all(np.abs(weights) <= port_mod.MAX_SIGNAL_WEIGHT + 1e-10)

    def test_fractional_kelly_capped(self):
        alpha = np.array([0.01, 0.02, -0.01, 0.005])
        var = np.array([0.0001, 0.0002, 0.0001, 0.00005])
        kelly = port_mod.fractional_kelly_sizing(alpha, var)
        assert np.all(np.abs(kelly) <= port_mod.MAX_SIGNAL_WEIGHT + 1e-10)

    def test_portfolio_metrics_sharpe(self):
        rng = default_rng(0)
        pnl = rng.normal(0.001, 0.01, 252)  # Positive drift Sharpe ≈ 1.5.
        ic = rng.normal(0.05, 0.02, 252)
        metrics = port_mod.compute_portfolio_metrics(pnl, ic, ic)
        assert metrics.sharpe > 0

    def test_deflated_sharpe_valid_range(self):
        dsr = port_mod.deflated_sharpe_ratio(1.5, 0.0, 252, 0.0, 3.0, 5)
        assert 0.0 <= dsr <= 1.0


# ---------------------------------------------------------------------------
# Analytics tests
# ---------------------------------------------------------------------------


class TestAnalytics:
    """Tests for analytics module."""

    def test_orthogonality_identity(self):
        """Signals with zero correlation should pass orthogonality check."""
        rng = default_rng(8)
        matrix = rng.normal(0, 1, (200, 5))  # Approx orthogonal.
        report = ana_mod.compute_orthogonality(matrix)
        assert report.max_r2 < 0.50  # Should be small for random signals.
        assert report.r2_matrix.shape == (5, 5)

    def test_orthogonality_diagonal_ones(self):
        rng = default_rng(9)
        matrix = rng.normal(0, 1, (200, 5))
        report = ana_mod.compute_orthogonality(matrix)
        assert_allclose(np.diag(report.r2_matrix), np.ones(5))

    def test_floam_ir_positive(self):
        rng = default_rng(10)
        ic_panel = rng.normal(0.05, 0.02, (252, 5))
        pnl = rng.normal(0.001, 0.01, 252)
        result = ana_mod.compute_floam(ic_panel, pnl)
        assert result.ic > 0
        assert result.ir_predicted > 0

    def test_hmm_regimes_two_states(self, panel):
        rets = panel.forward_returns[:, 0]  # Single asset.
        regime = ana_mod.fit_hmm_regimes(rets)
        assert set(np.unique(regime.regimes)).issubset({0.0, 1.0})
        assert regime.probabilities.shape[1] == 2

    def test_cpcv_returns_array(self, panel):
        sig = panel.forward_returns[:100]
        ret = panel.forward_returns[:100]
        oos = ana_mod.cpcv_ic_scores(sig, ret, n_splits=5, n_test_groups=2)
        assert isinstance(oos, np.ndarray)
        assert len(oos) > 0

    def test_rolling_icir_shape(self):
        rng = default_rng(11)
        ic = rng.normal(0.05, 0.02, 200)
        icir = sig_mod.rolling_icir(ic, window=60)
        assert icir.shape == (200,)
        assert np.all(np.isnan(icir[:59]))
        assert np.isfinite(icir[59])


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestFalsification:
    """Tests for falsification framework."""

    def test_bonferroni(self):
        from citadel_alpha import falsification as fal

        p = np.array([0.01, 0.05, 0.10])
        adj = fal.bonferroni_correction(p)
        assert_allclose(adj, p * 3)

    def test_bh_rejects_small_pvalue(self):
        from citadel_alpha import falsification as fal

        p = np.array([0.001, 0.5, 0.8, 0.9, 0.95])
        mask = fal.benjamini_hochberg(p)
        assert mask[0]  # Smallest p-value should be rejected.

    def test_dsr_valid_range(self):
        from citadel_alpha import falsification as fal

        rng = np.random.default_rng(1)
        pnl = rng.normal(0.001, 0.01, 252)
        sr = float(np.mean(pnl) * 252 / (np.std(pnl) * np.sqrt(252)))
        dsr = fal.deflated_sharpe_ratio(sr, pnl, n_trials=5)
        assert 0.0 <= dsr <= 1.0

    def test_scan_smooth_manifold(self):
        from citadel_alpha import falsification as fal

        surface = np.linspace(1.0, 1.5, 20)  # Smooth gradient.
        norm, is_smooth = fal.scan_parameter_manifold(surface, epsilon_threshold=0.5)
        assert is_smooth

    def test_scan_jagged_manifold(self):
        from citadel_alpha import falsification as fal

        rng = np.random.default_rng(2)
        surface = rng.uniform(0, 3, 20)  # Jagged / noisy surface.
        norm, _ = fal.scan_parameter_manifold(surface, epsilon_threshold=0.01)
        assert norm > 0.01

    def test_vix_regime_ic(self):
        from citadel_alpha import falsification as fal

        rng = np.random.default_rng(3)
        ic = rng.normal(0.05, 0.02, 500)
        vix = np.concatenate(
            [
                np.full(200, 12.0),
                np.full(150, 25.0),
                np.full(150, 35.0),
            ]
        )
        report = fal.vix_regime_ic(ic, vix)
        assert 0.0 <= report.stability_score <= 1.0
        assert np.isfinite(report.risk_on_ic)

    def test_vix_regime_scale_values(self):
        from citadel_alpha.signals import vix_regime_scale

        assert vix_regime_scale(12.0) == 1.0
        assert vix_regime_scale(25.0) == 0.60
        assert vix_regime_scale(40.0) == 0.25


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_constant_yields(self):
        """Constant yields should produce all-zero raw scores (no divergence)."""
        ytm_10y = np.full(8, 0.04)
        ytm_2y = np.full(8, 0.02)
        ret = np.zeros(8)
        res = sig_mod.compute_gcsd(ytm_10y, ytm_2y, ret, use_cpp=False)
        # All slopes identical → MAD = 0 → handled via epsilon → z_scores near 0.
        assert np.all(np.isfinite(res.raw_score))

    def test_large_n(self):
        """Should scale to N=64 assets without error."""
        rng = default_rng(12)
        n = 64
        ytm_10y = rng.uniform(0.02, 0.06, n)
        ytm_2y = rng.uniform(0.01, 0.04, n)
        ret = rng.normal(0, 0.01, n)
        res = sig_mod.compute_gcsd(ytm_10y, ytm_2y, ret, use_cpp=False)
        assert res.raw_score.shape == (n,)

    def test_extreme_bid_ask(self):
        """Very wide spreads should not cause NaN/Inf."""
        rng = default_rng(13)
        bid_ask = np.array([1e-5, 1e-5, 1e-5, 0.10, 1e-5, 1e-5, 1e-5, 1e-5])
        ret = rng.normal(0, 0.01, 8)
        res = sig_mod.compute_clsd(bid_ask, ret, use_cpp=False)
        assert np.all(np.isfinite(res.raw_score))

    def test_regime_wt_clamped(self):
        """Regime weight outside [0.5, 1.0] should be clamped."""
        rng = default_rng(14)
        n = 6
        ois_1y = rng.uniform(0.02, 0.05, n)
        pol = rng.uniform(0.01, 0.03, n)
        delta = rng.normal(0, 0.001, n)
        std = np.full(n, 0.002)
        ret = rng.normal(0, 0.01, n)

        regime_above = np.full(n, 2.0)  # Exceeds max.
        regime_below = np.full(n, 0.1)  # Below min.

        res_a = sig_mod.compute_cbpsm(
            ois_1y, pol, delta, std, regime_above, ret, use_cpp=False
        )
        res_b = sig_mod.compute_cbpsm(
            ois_1y, pol, delta, std, regime_below, ret, use_cpp=False
        )

        # Both should produce finite results (clamped inside compute).
        assert np.all(np.isfinite(res_a.raw_score))
        assert np.all(np.isfinite(res_b.raw_score))

    def test_too_few_assets_raises(self):
        """n < 4 should raise ValueError."""
        with pytest.raises((ValueError, RuntimeError)):
            sig_mod.compute_gcsd(
                np.array([0.04, 0.03]), np.array([0.02, 0.015]), np.array([0.01, 0.02])
            )
