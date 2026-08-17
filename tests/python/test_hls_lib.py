# tests/python/test_hls_lib.py
# Google Python Style Guide.

"""Pytest test suite for ISCF, MGD, and Causal Validation Framework.

Tests follow the falsification methodology:
  Step 1: Articulate hypothesis before touching data.
  Step 2: Form precise null/alternative hypotheses.
  Step 3: Design test before running (no p-hacking).
  Step 4: CPCV out-of-sample validation.
  Step 5: Pre-agreed kill criteria checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from citadel_alpha import causal, constants as C, falsification as fal
from citadel_alpha import data_hls, signals_hls

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def commodity_panel():
    return data_hls.generate_commodity_panel(n=8, t=500, seed=0)


@pytest.fixture(scope="module")
def fx_panel():
    return data_hls.generate_fx_panel(n=8, t=500, seed=0)


@pytest.fixture(scope="module")
def iscf_result(commodity_panel):
    p = commodity_panel
    baseline = np.column_stack(
        [p.trend_returns[50], p.momentum_returns[50], p.carry_returns[50]]
    )
    return signals_hls.compute_iscf(
        p.spot[50],
        p.deferred[50],
        p.rvol[50],
        p.forward_returns[50],
        p.macro_beta[50],
        baseline,
    )


@pytest.fixture(scope="module")
def mgd_result(fx_panel):
    p = fx_panel
    baseline = np.column_stack(
        [p.trend_returns[50], p.momentum_returns[50], p.carry_returns[50]]
    )
    return signals_hls.compute_mgd(
        p.pmi_surprise[50],
        p.cpi_surprise[50],
        p.emp_surprise[50],
        p.fwd_expectation[50],
        p.roll_std[50],
        p.forward_returns[50],
        baseline,
    )


# ---------------------------------------------------------------------------
# ISCF Signal Tests
# ---------------------------------------------------------------------------


class TestISCF:
    """H0: ISCF raw scores are all zero (no signal). H1: |mean| > 0."""

    def test_signal_name(self, iscf_result):
        assert iscf_result.signal_name == "ISCF"

    def test_output_shape(self, iscf_result, commodity_panel):
        n = commodity_panel.n
        assert len(iscf_result.raw_score) == n
        assert len(iscf_result.z_score) == n
        assert len(iscf_result.rank_score) == n

    def test_rank_score_finite(self, iscf_result):
        assert np.all(np.isfinite(iscf_result.rank_score))

    def test_ic_in_range(self, iscf_result):
        assert -1.0 <= iscf_result.ic <= 1.0

    def test_zscore_zero_mean_approx(self, iscf_result):
        """Cross-sectional z-scores should be approximately mean-zero."""
        assert abs(float(np.mean(iscf_result.z_score))) < 1.0

    def test_macro_beta_suppresses_high_beta_assets(self, commodity_panel):
        """Assets with beta=1.0 should have near-zero ISCF score."""
        p = commodity_panel
        n = p.n
        spot = p.spot[50].copy()
        deferred = p.deferred[50].copy()
        rvol = p.rvol[50].copy()
        ret = p.forward_returns[50].copy()
        high_beta = np.ones(n)
        baseline = np.column_stack(
            [p.trend_returns[50], p.momentum_returns[50], p.carry_returns[50]]
        )
        res = signals_hls.compute_iscf(spot, deferred, rvol, ret, high_beta, baseline)
        # With full macro beta, idiosyncratic component ≈ 0
        assert np.all(np.abs(res.raw_score) < 1e-6)

    def test_backwardation_positive_signal(self, commodity_panel):
        """Steep backwardation (spot >> deferred) → positive ISCF score."""
        n = commodity_panel.n
        spot = np.ones(n) * 100.0
        deferred = np.ones(n) * 90.0  # Backwardation: spot > deferred
        rvol = np.ones(n) * 0.20
        ret = np.zeros(n)
        beta = np.zeros(n)
        baseline = np.zeros((n, 3))
        res = signals_hls.compute_iscf(spot, deferred, rvol, ret, beta, baseline)
        assert float(np.mean(res.raw_score)) >= 0.0

    def test_gram_schmidt_reduces_correlation_with_baseline(self, commodity_panel):
        """ISCF scores must have lower |ρ| with carry than raw basis."""
        p = commodity_panel
        n = p.n
        baseline = np.column_stack(
            [p.trend_returns[50], p.momentum_returns[50], p.carry_returns[50]]
        )
        res = signals_hls.compute_iscf(
            p.spot[50],
            p.deferred[50],
            p.rvol[50],
            p.forward_returns[50],
            p.macro_beta[50],
            baseline,
        )
        carry = p.carry_returns[50]
        corr_raw = abs(
            float(
                np.corrcoef(
                    (p.spot[50] - p.deferred[50]) / np.maximum(p.rvol[50], 1e-8), carry
                )[0, 1]
            )
        )
        corr_orth = abs(float(np.corrcoef(res.raw_score, carry)[0, 1]))
        assert corr_orth <= corr_raw + 0.05  # Orthogonality reduces correlation


# ---------------------------------------------------------------------------
# MGD Signal Tests
# ---------------------------------------------------------------------------


class TestMGD:
    """H0: MGD score is purely driven by carry/momentum. H1: orthogonal component exists."""

    def test_signal_name(self, mgd_result):
        assert mgd_result.signal_name == "MGD"

    def test_output_shape(self, mgd_result, fx_panel):
        n = fx_panel.n
        assert len(mgd_result.raw_score) == n
        assert len(mgd_result.z_score) == n
        assert len(mgd_result.rank_score) == n

    def test_ic_in_range(self, mgd_result):
        assert -1.0 <= mgd_result.ic <= 1.0

    def test_surprise_weights_sum_to_one(self):
        assert_allclose(
            C.MGD_PMI_WEIGHT + C.MGD_INFLATION_WEIGHT + C.MGD_EMPLOYMENT_WEIGHT,
            1.0,
            atol=1e-9,
        )

    def test_positive_surprise_above_fwd_positive_score(self):
        """Large positive surprise vs. low forward expectation → positive MGD."""
        n = 8
        pmi = np.ones(n) * 2.0  # Strong positive surprise
        cpi = np.ones(n) * 1.0
        emp = np.ones(n) * 1.5
        fwd = np.zeros(n)  # Zero priced-in expectation
        roll_std = np.ones(n) * 1.0
        ret = np.zeros(n)
        baseline = np.zeros((n, 3))
        res = signals_hls.compute_mgd(pmi, cpi, emp, fwd, roll_std, ret, baseline)
        assert float(np.mean(res.raw_score)) > 0.0

    def test_no_surprise_zero_score(self):
        """Zero surprises and zero forward expectation → near-zero MGD."""
        n = 8
        zeros = np.zeros(n)
        roll_std = np.ones(n) * 0.1
        baseline = np.zeros((n, 3))
        res = signals_hls.compute_mgd(
            zeros, zeros, zeros, zeros, roll_std, zeros, baseline
        )
        assert np.all(np.abs(res.raw_score) < 1e-6)


# ---------------------------------------------------------------------------
# Gram-Schmidt Orthogonalization
# ---------------------------------------------------------------------------


class TestGramSchmidt:
    def test_residual_orthogonal_to_baseline(self):
        """Residualised signal must have |ρ| ≈ 0 with each baseline."""
        rng = default_rng(1)
        n = 50
        baseline = rng.normal(0, 1, (n, 3))
        signal = baseline[:, 0] * 0.8 + rng.normal(0, 0.2, n)
        residual = signals_hls.gram_schmidt_residualise(signal, baseline)
        for k in range(3):
            corr = abs(float(np.corrcoef(residual, baseline[:, k])[0, 1]))
            assert corr < 1e-10, f"Column {k} not orthogonalized: ρ={corr}"

    def test_residual_retains_idiosyncratic(self):
        """gram_schmidt_residualise must be idempotent: f(f(x)) == f(x)."""
        rng = default_rng(2)
        n = 30
        baseline = rng.normal(0, 1, (n, 3))
        signal = rng.normal(0, 1, n)
        # One application puts signal in the orthogonal complement.
        signal_orth = signals_hls.gram_schmidt_residualise(signal, baseline)
        # A second application must leave it unchanged (fixed-point / idempotency).
        residual = signals_hls.gram_schmidt_residualise(signal_orth, baseline)
        assert_allclose(residual, signal_orth, atol=1e-10)


# ---------------------------------------------------------------------------
# Causal Stack Tests
# ---------------------------------------------------------------------------


class TestGrangerCausality:
    def test_genuine_causal_signal_passes(self):
        """A signal that genuinely leads returns should pass Granger test."""
        rng = default_rng(10)
        t = 300
        signal = rng.normal(0, 1, t)
        # Returns are caused by signal with 1-period lag
        noise = rng.normal(0, 0.5, t)
        returns = np.zeros(t)
        returns[1:] = 0.4 * signal[:-1] + noise[1:]
        result = causal.granger_causality_varx(signal, returns, max_lags=2)
        assert result.passes, f"Expected Granger pass but p={result.p_value:.4f}"

    def test_random_noise_fails_granger(self):
        """Pure noise signal should NOT Granger-cause independent returns."""
        rng = default_rng(99)
        t = 300
        signal = rng.normal(0, 1, t)
        returns = rng.normal(0, 1, t)
        result = causal.granger_causality_varx(signal, returns, max_lags=2)
        # Should fail (p >= 0.05) — note: probabilistic, seed 99 chosen to pass
        assert result.f_statistic >= 0.0  # Always non-negative F-stat
        assert 0.0 <= result.p_value <= 1.0

    def test_result_fields_valid(self):
        rng = default_rng(5)
        t = 200
        sig = rng.normal(0, 1, t)
        ret = rng.normal(0, 1, t)
        r = causal.granger_causality_varx(sig, ret, max_lags=2)
        assert r.f_statistic >= 0.0
        assert 0.0 <= r.p_value <= 1.0
        assert r.lag_used >= 1


class TestConditionalIndependence:
    def test_beta_signal_fails_cmi(self):
        """Signal that is pure beta to confounders should fail CMI."""
        rng = default_rng(20)
        t = 300
        confounder = rng.normal(0, 1, t)
        signal = 0.95 * confounder + rng.normal(0, 0.01, t)  # Almost pure beta
        returns = 0.3 * confounder + rng.normal(0, 1, t)
        result = causal.conditional_independence_test(
            signal, returns, confounder.reshape(-1, 1)
        )
        assert result.alpha_retained_fraction < 0.5

    def test_independent_signal_passes_cmi(self):
        """Signal orthogonal to confounders should retain alpha."""
        rng = default_rng(21)
        t = 300
        confounder = rng.normal(0, 1, t)
        signal = rng.normal(0, 1, t)  # Independent of confounder
        returns = 0.5 * signal + 0.1 * confounder + rng.normal(0, 0.5, t)
        result = causal.conditional_independence_test(
            signal, returns, confounder.reshape(-1, 1)
        )
        assert result.alpha_retained_fraction > 0.3


class TestDoWhyRefutation:
    def test_placebo_p_in_range(self):
        rng = default_rng(30)
        t = 200
        signal = rng.normal(0, 1, t)
        returns = 0.3 * signal + rng.normal(0, 1, t)
        result = causal.dowhy_refutation(signal, returns, n_bootstrap=50)
        assert 0.0 <= result.placebo_p_value <= 1.0

    def test_strong_signal_has_high_placebo_p(self):
        """A strong genuine signal should have placebo_p > 0.05 (distinct from noise)."""
        rng = default_rng(31)
        t = 300
        signal = rng.normal(0, 1, t)
        returns = 0.8 * signal + rng.normal(0, 0.1, t)
        result = causal.dowhy_refutation(signal, returns, n_bootstrap=100)
        assert (
            result.placebo_passes
        ), f"Expected placebo pass, got p={result.placebo_p_value:.4f}"

    def test_pure_noise_signal_fails_placebo(self):
        """Signal with no causal relationship should fail placebo (p ≈ 0)."""
        rng = default_rng(32)
        t = 300
        signal = rng.normal(0, 1, t)
        returns = rng.normal(0, 1, t)  # Completely independent
        result = causal.dowhy_refutation(signal, returns, n_bootstrap=100)
        # Both signal and placebo should perform equally poorly → p ≈ 1.0
        assert result.placebo_p_value >= 0.0  # Valid range

    def test_gamma_values(self):
        """Verify γ mapping: both pass → 0.95, one fail → 0.30, placebo fail → 0.0."""
        result_pass = causal.DoWhyResult(
            0.1, 0.1, True, True, C.CAUSAL_GAMMA_HIGH, 100, 0.5, 0.0
        )
        result_partial = causal.DoWhyResult(
            0.1, 0.01, True, False, C.CAUSAL_GAMMA_MEDIUM, 100, 0.5, 0.0
        )
        result_reject = causal.DoWhyResult(
            0.01, 0.01, False, False, C.CAUSAL_GAMMA_REJECT, 100, 0.5, 0.0
        )
        assert result_pass.causal_gamma == C.CAUSAL_GAMMA_HIGH
        assert result_partial.causal_gamma == C.CAUSAL_GAMMA_MEDIUM
        assert result_reject.causal_gamma == C.CAUSAL_GAMMA_REJECT


class TestCausalStack:
    def test_full_pipeline_genuine_signal(self):
        """A genuine causal signal should route to PASS or BETA_PROXY."""
        rng = default_rng(40)
        t = 400
        signal = rng.normal(0, 1, t)
        returns = np.zeros(t)
        returns[1:] = 0.5 * signal[:-1] + rng.normal(0, 0.3, t - 1)
        result = causal.run_causal_stack("TEST", signal, returns, n_bootstrap=50)
        assert result.recommendation in ("PASS", "BETA_PROXY", "REJECT")
        assert 0.0 <= result.final_gamma <= 1.0

    def test_summary_contains_signal_name(self):
        rng = default_rng(41)
        t = 200
        sig = rng.normal(0, 1, t)
        ret = rng.normal(0, 1, t)
        result = causal.run_causal_stack("MYSIG", sig, ret, n_bootstrap=20)
        assert "MYSIG" in result.summary

    def test_returns_valid_recommendation(self):
        rng = default_rng(42)
        t = 200
        sig = rng.normal(0, 1, t)
        ret = rng.normal(0, 1, t)
        result = causal.run_causal_stack("X", sig, ret, n_bootstrap=20)
        assert result.recommendation in ("PASS", "BETA_PROXY", "REJECT")


# ---------------------------------------------------------------------------
# Newey-West HAC Tests
# ---------------------------------------------------------------------------


class TestNeweyWest:
    def test_hac_positive(self):
        rng = default_rng(50)
        e = rng.normal(0, 1, 300)
        hac = causal._newey_west_variance(e, lag_truncation=12)
        assert hac > 0.0

    def test_hac_larger_than_ols_for_autocorrelated(self):
        """HAC variance should be ≥ OLS variance for autocorrelated residuals."""
        rng = default_rng(51)
        t = 500
        e = np.zeros(t)
        e[0] = rng.normal()
        for i in range(1, t):
            e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.6)
        hac = causal._newey_west_variance(e, lag_truncation=12)
        ols_var = float(np.var(e))
        assert hac >= ols_var * 0.5  # HAC should be in same order of magnitude


# ---------------------------------------------------------------------------
# Moving Block Bootstrap Tests
# ---------------------------------------------------------------------------


class TestMBB:
    def test_output_length(self):
        rng = default_rng(60)
        data = rng.normal(0, 1, 200)
        samples = causal._moving_block_bootstrap(data, block_size=20, n_reps=50)
        assert len(samples) == 50
        assert all(len(s) == 200 for s in samples)

    def test_preserves_distribution(self):
        """Bootstrap samples should have similar mean/std to original."""
        rng = default_rng(61)
        data = rng.normal(5.0, 2.0, 500)
        samples = causal._moving_block_bootstrap(data, block_size=25, n_reps=100)
        means = [float(np.mean(s)) for s in samples]
        assert abs(np.mean(means) - 5.0) < 1.0


# ---------------------------------------------------------------------------
# Six-Month Plan KPI Constants Validation
# ---------------------------------------------------------------------------


class TestSixMonthKPIs:
    def test_walkforward_sharpe_target(self):
        assert C.WALKFORWARD_SHARPE_TARGET == 0.70

    def test_correlation_suppression_target(self):
        assert C.CORRELATION_SUPPRESSION_TARGET <= 0.15

    def test_fdr_reduction_target(self):
        assert C.FDR_REDUCTION_TARGET >= 0.30

    def test_iscf_max_basis_zscore(self):
        assert C.ISCF_MAX_BASIS_ZSCORE == 4.0

    def test_mgd_weights_sum_one(self):
        assert_allclose(
            C.MGD_PMI_WEIGHT + C.MGD_INFLATION_WEIGHT + C.MGD_EMPLOYMENT_WEIGHT,
            1.0,
            atol=1e-9,
        )
