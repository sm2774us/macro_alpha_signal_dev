// tests/cpp/test_cpp_lib.cpp
// Google C++ Style Guide. Google Test.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "alpha_engine.hpp"

namespace ca = citadel::alpha;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<double> Linspace(double lo, double hi, std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
  return v;
}

std::vector<double> Uniform(double lo, double hi, std::size_t n, unsigned seed = 42) {
  std::vector<double> v(n);
  // Simple LCG for test reproducibility (no <random> overhead).
  uint64_t s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double t = static_cast<double>(s >> 11) / (1ULL << 53);
    v[i] = lo + (hi - lo) * t;
  }
  return v;
}

// ---------------------------------------------------------------------------
// Utility tests
// ---------------------------------------------------------------------------

TEST(UtilityTest, MedianAbsDevSorted) {
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  double mad = ca::MedianAbsDev(x);
  // Median = 3, deviations = {2,1,0,1,2}, MAD = 1.
  EXPECT_NEAR(mad, 1.0, 1e-10);
}

TEST(UtilityTest, MedianAbsDevEven) {
  std::vector<double> x = {1.0, 3.0, 5.0, 7.0};
  double mad = ca::MedianAbsDev(x);
  // Median = 4, deviations = {3,1,1,3}, MAD = 2.
  EXPECT_NEAR(mad, 2.0, 1e-10);
}

TEST(UtilityTest, GaussianRankNormalizeMonotone) {
  std::vector<double> v = {5.0, 1.0, 3.0, 2.0, 4.0};
  auto orig_sorted = v;
  std::sort(orig_sorted.begin(), orig_sorted.end());
  ca::GaussianRankNormalize(v);
  // Check values at sorted positions are in ascending order.
  std::vector<double> sorted_result(v.begin(), v.end());
  std::sort(sorted_result.begin(), sorted_result.end());
  for (std::size_t i = 1; i < sorted_result.size(); ++i) {
    EXPECT_LT(sorted_result[i - 1], sorted_result[i]);
  }
}

TEST(UtilityTest, GaussianRankNormalizeAllFinite) {
  auto v = Uniform(-5.0, 5.0, 16, 7);
  ca::GaussianRankNormalize(v);
  for (double x : v) EXPECT_TRUE(std::isfinite(x));
}

TEST(UtilityTest, ComputeICPerfectCorrelation) {
  std::vector<double> sig = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> ret = {1.0, 2.0, 3.0, 4.0, 5.0};
  double ic = ca::ComputeIC(sig, ret);
  EXPECT_NEAR(ic, 1.0, 1e-10);
}

TEST(UtilityTest, ComputeICPerfectAnticorrelation) {
  std::vector<double> sig = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> ret = {5.0, 4.0, 3.0, 2.0, 1.0};
  double ic = ca::ComputeIC(sig, ret);
  EXPECT_NEAR(ic, -1.0, 1e-10);
}

TEST(UtilityTest, ComputeICTooFew) {
  std::vector<double> sig = {1.0, 2.0};
  std::vector<double> ret = {1.0, 2.0};
  EXPECT_EQ(ca::ComputeIC(sig, ret), 0.0);
}

TEST(UtilityTest, LedoitWolfRetainsPositiveDiag) {
  const std::size_t n = 3;
  // Simple diagonal matrix.
  std::vector<double> cov = {
      4.0, 0.0, 0.0,
      0.0, 4.0, 0.0,
      0.0, 0.0, 4.0,
  };
  ca::LedoitWolfShrink(cov, n);
  // Shrunk matrix should still have positive diagonal.
  EXPECT_GT(cov[0], 0.0);
  EXPECT_GT(cov[4], 0.0);
  EXPECT_GT(cov[8], 0.0);
}

TEST(UtilityTest, DeflatedSharpeInRange) {
  double dsr = ca::DeflatedSharpeRatio(1.5, 0.0, 252, 0.0, 3.0, 5);
  EXPECT_GE(dsr, 0.0);
  EXPECT_LE(dsr, 1.0);
}

TEST(UtilityTest, EMAUpdateCorrect) {
  double ema = ca::EMAUpdate(10.0, 20.0, 0.1);
  EXPECT_NEAR(ema, 0.1 * 20.0 + 0.9 * 10.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Signal 1 — GCSD
// ---------------------------------------------------------------------------

TEST(GCSDTest, OutputShape) {
  const std::size_t n = 8;
  auto ytm_10y = Uniform(0.02, 0.06, n);
  auto ytm_2y = Uniform(0.01, 0.03, n);
  auto ret = Uniform(-0.01, 0.01, n, 1);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  EXPECT_EQ(result.raw_score.size(), n);
  EXPECT_EQ(result.z_score.size(), n);
  EXPECT_EQ(result.rank_score.size(), n);
}

TEST(GCSDTest, ZScoreMeanNearZero) {
  const std::size_t n = 10;
  auto ytm_10y = Uniform(0.02, 0.06, n);
  auto ytm_2y = Uniform(0.01, 0.03, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  double mean_z = 0.0;
  for (double v : result.z_score) mean_z += v;
  mean_z /= static_cast<double>(n);
  EXPECT_NEAR(mean_z, 0.0, 1e-10);
}

TEST(GCSDTest, ICInValidRange) {
  const std::size_t n = 10;
  auto ytm_10y = Uniform(0.02, 0.06, n);
  auto ytm_2y = Uniform(0.01, 0.03, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  EXPECT_GE(result.ic, -1.0);
  EXPECT_LE(result.ic, 1.0);
}

TEST(GCSDTest, AllFinite) {
  const std::size_t n = 12;
  auto ytm_10y = Uniform(0.02, 0.06, n);
  auto ytm_2y = Uniform(0.01, 0.03, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
  for (double v : result.z_score) EXPECT_TRUE(std::isfinite(v));
  for (double v : result.rank_score) EXPECT_TRUE(std::isfinite(v));
}

TEST(GCSDTest, SizeMismatchThrows) {
  auto ytm_10y = Uniform(0.02, 0.06, 5);
  auto ytm_2y = Uniform(0.01, 0.03, 8);
  auto ret = Uniform(-0.01, 0.01, 8);
  EXPECT_THROW(
      ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret),
      std::invalid_argument);
}

TEST(GCSDTest, SignalNameCorrect) {
  const std::size_t n = 6;
  auto ytm_10y = Uniform(0.02, 0.06, n);
  auto ytm_2y = Uniform(0.01, 0.03, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  EXPECT_EQ(result.signal_name, "GCSD");
}

// ---------------------------------------------------------------------------
// Signal 2 — RVIS
// ---------------------------------------------------------------------------

TEST(RVISTest, BasicValidity) {
  const std::size_t n = 10;
  auto iv = Uniform(0.10, 0.30, n);
  auto rv = Uniform(0.08, 0.20, n, 2);
  auto vrp_std = Uniform(0.01, 0.05, n, 3);
  auto ret = Uniform(-0.01, 0.01, n, 4);
  auto result = ca::RealisedImpliedVolSpread::Compute(iv, rv, vrp_std, ret);
  EXPECT_EQ(result.raw_score.size(), n);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
  EXPECT_GE(result.ic, -1.0);
  EXPECT_LE(result.ic, 1.0);
}

TEST(RVISTest, SignalNameCorrect) {
  const std::size_t n = 6;
  auto iv = Uniform(0.10, 0.30, n);
  auto rv = Uniform(0.08, 0.20, n);
  auto vrp_std = Uniform(0.01, 0.05, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::RealisedImpliedVolSpread::Compute(iv, rv, vrp_std, ret);
  EXPECT_EQ(result.signal_name, "RVIS");
}

// ---------------------------------------------------------------------------
// Signal 3 — MSDI
// ---------------------------------------------------------------------------

TEST(MSDITest, BasicValidity) {
  const std::size_t n = 10;
  auto surprises = Uniform(-2.0, 2.0, n);
  auto ema_prev = Uniform(-1.0, 1.0, n, 5);
  auto rolling_std = Uniform(0.5, 2.0, n, 6);
  auto ret = Uniform(-0.01, 0.01, n, 7);
  double ema_alpha = 2.0 / 11.0;
  auto result = ca::MacroSurpriseDiffusionIndex::Compute(
      surprises, ema_prev, ema_alpha, rolling_std, ret);
  EXPECT_EQ(result.raw_score.size(), n);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
  EXPECT_EQ(result.signal_name, "MSDI");
}

// ---------------------------------------------------------------------------
// Signal 4 — CLSD
// ---------------------------------------------------------------------------

TEST(CLSDTest, HighestSpreadLowestScore) {
  // Construct spreads where the last element is very wide.
  std::vector<double> bid_ask = {0.0001, 0.0002, 0.0001, 0.0003,
                                  0.0001, 0.0001, 0.0001, 0.1000};
  std::vector<double> ret(8, 0.0);
  auto result = ca::CrossAssetLiquidityStressDivergence::Compute(bid_ask, ret);
  // Index 7 has the highest spread → should have the lowest (most negative) raw score.
  double min_raw = *std::min_element(result.raw_score.begin(), result.raw_score.end());
  EXPECT_NEAR(result.raw_score[7], min_raw, 1e-10);
}

TEST(CLSDTest, AllFinite) {
  auto bid_ask = Uniform(0.0001, 0.005, 10);
  auto ret = Uniform(-0.01, 0.01, 10);
  auto result = ca::CrossAssetLiquidityStressDivergence::Compute(bid_ask, ret);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Signal 5 — CBPSM
// ---------------------------------------------------------------------------

TEST(CBPSMTest, RegimeWeightScalesOutput) {
  const std::size_t n = 8;
  auto ois = Uniform(0.02, 0.05, n);
  auto pol = Uniform(0.01, 0.03, n, 10);
  auto delta = Uniform(-0.002, 0.002, n, 11);
  auto std_v = Uniform(0.001, 0.005, n, 12);
  auto ret = Uniform(-0.01, 0.01, n, 13);

  std::vector<double> regime_on(n, 1.0);
  std::vector<double> regime_off(n, 0.5);

  auto result_on = ca::CentralBankPolicySurpriseMomentum::Compute(
      ois, pol, delta, std_v, regime_on, ret);
  auto result_off = ca::CentralBankPolicySurpriseMomentum::Compute(
      ois, pol, delta, std_v, regime_off, ret);

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_GE(std::abs(result_on.raw_score[i]),
              std::abs(result_off.raw_score[i]) - 1e-10);
  }
}

TEST(CBPSMTest, SignalNameCorrect) {
  const std::size_t n = 6;
  auto ois = Uniform(0.02, 0.05, n);
  auto pol = Uniform(0.01, 0.03, n);
  auto delta = Uniform(-0.002, 0.002, n);
  auto std_v = Uniform(0.001, 0.005, n);
  auto regime = Uniform(0.5, 1.0, n);
  auto ret = Uniform(-0.01, 0.01, n);
  auto result = ca::CentralBankPolicySurpriseMomentum::Compute(
      ois, pol, delta, std_v, regime, ret);
  EXPECT_EQ(result.signal_name, "CBPSM");
}

// ---------------------------------------------------------------------------
// Stress tests
// ---------------------------------------------------------------------------

TEST(StressTest, LargeNAssets) {
  const std::size_t n = 64;
  auto ytm_10y = Uniform(0.01, 0.10, n);
  auto ytm_2y = Uniform(0.005, 0.06, n, 20);
  auto ret = Uniform(-0.02, 0.02, n, 21);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  EXPECT_EQ(result.raw_score.size(), n);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
}

TEST(StressTest, ExtremeInputValues) {
  // Near-zero yields.
  std::vector<double> ytm_10y(8, 1e-4);
  std::vector<double> ytm_2y(8, 1e-5);
  std::vector<double> ret(8, 0.0);
  auto result = ca::GlobalYieldCurveSlopeDivergence::Compute(ytm_10y, ytm_2y, ret);
  for (double v : result.raw_score) EXPECT_TRUE(std::isfinite(v));
}
