// Copyright 2025 Citadel Systematic Macro Pod
// alpha_engine.cpp — C++26 hot-path implementation.
// Google C++ Style Guide. Data-oriented design (SoA). Cache-friendly.

#include "alpha_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace citadel::alpha {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

// Returns sorted copy of indices by value (ascending).
std::vector<std::size_t> ArgSort(std::span<const double> x) {
  std::vector<std::size_t> idx(x.size());
  std::iota(idx.begin(), idx.end(), 0u);
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  return idx;
}

// Computes Spearman rank correlation between two equal-length vectors.
double SpearmanCorr(std::span<const double> x, std::span<const double> y) {
  const std::size_t n = x.size();
  if (n < 4) return 0.0;

  auto rank_vec = [n](std::span<const double> v) -> std::vector<double> {
    auto idx = ArgSort(v);
    std::vector<double> ranks(n);
    for (std::size_t i = 0; i < n; ++i) ranks[idx[i]] = static_cast<double>(i);
    return ranks;
  };

  auto rx = rank_vec(x);
  auto ry = rank_vec(y);

  double sum_d2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = rx[i] - ry[i];
    sum_d2 += d * d;
  }
  double nd = static_cast<double>(n);
  return 1.0 - 6.0 * sum_d2 / (nd * (nd * nd - 1.0));
}

// Cross-sectional z-score (subtract mean, divide by std).
std::vector<double> CrossSectionalZScore(std::span<const double> x) {
  const std::size_t n = x.size();
  std::vector<double> z(n);
  double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
  double var = 0.0;
  for (auto v : x) var += (v - mean) * (v - mean);
  var /= static_cast<double>(n);
  double sd = std::sqrt(std::max(var, kMinVariance));
  for (std::size_t i = 0; i < n; ++i) z[i] = (x[i] - mean) / sd;
  return z;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public utilities
// ---------------------------------------------------------------------------

double MedianAbsDev(std::span<const double> x) {
  const std::size_t n = x.size();
  std::vector<double> sorted(x.begin(), x.end());
  std::sort(sorted.begin(), sorted.end());
  double median = (n % 2 == 0)
      ? 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
      : sorted[n / 2];
  std::vector<double> abs_dev(n);
  for (std::size_t i = 0; i < n; ++i) abs_dev[i] = std::abs(sorted[i] - median);
  std::sort(abs_dev.begin(), abs_dev.end());
  return (n % 2 == 0)
      ? 0.5 * (abs_dev[n / 2 - 1] + abs_dev[n / 2])
      : abs_dev[n / 2];
}

double ComputeIC(std::span<const double> signal,
                 std::span<const double> returns) {
  if (signal.size() != returns.size() || signal.size() < 4) return 0.0;
  return SpearmanCorr(signal, returns);
}

void GaussianRankNormalize(std::span<double> values) {
  const std::size_t n = values.size();
  auto idx = ArgSort(values);
  const double nd = static_cast<double>(n);
  // Blom's formula: Φ^{-1}((rank - 3/8) / (n + 1/4))
  // We approximate Φ^{-1} via a fast rational approximation.
  auto phi_inv = [](double p) -> double {
    // Abramowitz & Stegun 26.2.23 rational approximation, max error 4.5e-4
    constexpr double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    constexpr double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double t = std::sqrt(-2.0 * std::log(p <= 0.5 ? p : 1.0 - p));
    double num = c0 + t * (c1 + t * c2);
    double den = 1.0 + t * (d1 + t * (d2 + t * d3));
    double z = t - num / den;
    return p <= 0.5 ? -z : z;
  };
  for (std::size_t i = 0; i < n; ++i) {
    double p = (static_cast<double>(i) + 1.0 - 0.375) / (nd + 0.25);
    p = std::clamp(p, 1e-10, 1.0 - 1e-10);
    values[idx[i]] = phi_inv(p);
  }
}

void LedoitWolfShrink(std::span<double> cov_matrix, std::size_t n) {
  // Oracle approximating shrinkage (Ledoit-Wolf analytical formula).
  // Target: scaled identity F = mu * I where mu = trace(S)/n.
  if (n == 0 || cov_matrix.size() != n * n) return;

  double trace = 0.0;
  double frobenius_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    trace += cov_matrix[i * n + i];
    for (std::size_t j = 0; j < n; ++j) {
      double v = cov_matrix[i * n + j];
      frobenius_sq += v * v;
    }
  }

  double mu = trace / static_cast<double>(n);
  // delta: shrinkage intensity (simplified Ledoit-Wolf, assuming T >> N).
  // Full Oracle requires T; here we use the structural formula.
  // For production, T should be passed and used in the analytical estimator.
  double alpha_num = frobenius_sq + trace * trace;
  double alpha_den = frobenius_sq * static_cast<double>(n);
  double delta = (alpha_den > kMinVariance) ? (1.0 - alpha_num / alpha_den) : 0.0;
  delta = std::clamp(delta, 0.0, 1.0);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double target = (i == j) ? mu : 0.0;
      cov_matrix[i * n + j] =
          delta * target + (1.0 - delta) * cov_matrix[i * n + j];
    }
  }
}

double DeflatedSharpeRatio(double sr, double sr_benchmark, std::size_t T,
                            double skew, double kurt, std::size_t N_trials) {
  // Bailey & Lopez de Prado (2014): DSR = PSR(SR^*) - PSR with multiple test
  // correction.  PSR(SR^*) = Φ[ (SR - SR^*) * sqrt(T-1) / sqrt(1 - skew*SR +
  // (kurt-1)/4 * SR^2) ]
  // SR^* is adjusted for number of trials via Expected Maximum Sharpe.
  // Expected Max SR under Gaussian: SR^* ≈ (1-γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e))
  // Simplified approximation:
  auto phi = [](double z) -> double {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
  };

  double nd = static_cast<double>(N_trials);
  // Expected max Sharpe (under iid) adjusted for N_trials:
  double sr_star = sr_benchmark > 0.0 ? sr_benchmark
      : std::sqrt(2.0) * std::log(nd) / std::sqrt(std::log(nd * std::log(nd)));

  double td = static_cast<double>(T);
  double denom = std::sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr);
  denom = std::max(denom, kMinVariance);
  double z = (sr - sr_star) * std::sqrt(td - 1.0) / denom;
  return phi(z);
}

// ---------------------------------------------------------------------------
// Signal 1 — GCSD
// ---------------------------------------------------------------------------
SignalResult GlobalYieldCurveSlopeDivergence::Compute(
    std::span<const double> ytm_10y,
    std::span<const double> ytm_2y,
    std::span<const double> next_ret) {
  const std::size_t n = ytm_10y.size();
  if (ytm_2y.size() != n || next_ret.size() != n || n < 4) {
    throw std::invalid_argument("GCSD: input size mismatch or n < 4");
  }

  // Compute yield curve slope per asset.
  std::vector<double> slope(n);
  for (std::size_t i = 0; i < n; ++i) {
    slope[i] = std::log(std::max(ytm_10y[i], 1e-6)) -
               std::log(std::max(ytm_2y[i], 1e-6));
  }

  // Robust cross-sectional normalisation: (slope - median) / MAD.
  std::vector<double> sorted_slope(slope);
  std::sort(sorted_slope.begin(), sorted_slope.end());
  double med = (n % 2 == 0)
      ? 0.5 * (sorted_slope[n / 2 - 1] + sorted_slope[n / 2])
      : sorted_slope[n / 2];
  double mad = MedianAbsDev(slope);
  mad = std::max(mad, kMinVariance);

  std::vector<double> raw(n);
  for (std::size_t i = 0; i < n; ++i) raw[i] = (slope[i] - med) / mad;

  auto z = CrossSectionalZScore(raw);
  auto rank = z;
  GaussianRankNormalize(rank);

  double ic = ComputeIC(raw, next_ret);

  SignalResult result;
  result.raw_score = std::move(raw);
  result.z_score = std::move(z);
  result.rank_score = std::move(rank);
  result.ic = ic;
  result.icir = ic;  // Single-period; ICIR computed externally over rolling IC.
  result.signal_name = kName;
  return result;
}

// ---------------------------------------------------------------------------
// Signal 2 — RVIS
// ---------------------------------------------------------------------------
SignalResult RealisedImpliedVolSpread::Compute(
    std::span<const double> iv_current,
    std::span<const double> rv_current,
    std::span<const double> vrp_rolling_std,
    std::span<const double> next_ret) {
  const std::size_t n = iv_current.size();
  if (rv_current.size() != n || vrp_rolling_std.size() != n ||
      next_ret.size() != n || n < 4) {
    throw std::invalid_argument("RVIS: input size mismatch or n < 4");
  }

  std::vector<double> raw(n);
  for (std::size_t i = 0; i < n; ++i) {
    double vrp = iv_current[i] - rv_current[i];
    raw[i] = vrp / std::max(vrp_rolling_std[i], kMinVariance);
  }

  auto z = CrossSectionalZScore(raw);
  auto rank = z;
  GaussianRankNormalize(rank);

  SignalResult result;
  result.raw_score = std::move(raw);
  result.z_score = std::move(z);
  result.rank_score = std::move(rank);
  result.ic = ComputeIC(result.raw_score, next_ret);
  result.icir = result.ic;
  result.signal_name = kName;
  return result;
}

// ---------------------------------------------------------------------------
// Signal 3 — MSDI
// ---------------------------------------------------------------------------
SignalResult MacroSurpriseDiffusionIndex::Compute(
    std::span<const double> surprises,
    std::span<const double> ema_prev,
    double ema_alpha,
    std::span<const double> rolling_std,
    std::span<const double> next_ret) {
  const std::size_t n = surprises.size();
  if (ema_prev.size() != n || rolling_std.size() != n ||
      next_ret.size() != n || n < 4) {
    throw std::invalid_argument("MSDI: input size mismatch or n < 4");
  }

  std::vector<double> raw(n);
  for (std::size_t i = 0; i < n; ++i) {
    double ema_new = EMAUpdate(ema_prev[i], surprises[i], ema_alpha);
    raw[i] = ema_new / std::max(rolling_std[i], kMinVariance);
  }

  auto z = CrossSectionalZScore(raw);
  auto rank = z;
  GaussianRankNormalize(rank);

  SignalResult result;
  result.raw_score = std::move(raw);
  result.z_score = std::move(z);
  result.rank_score = std::move(rank);
  result.ic = ComputeIC(result.raw_score, next_ret);
  result.icir = result.ic;
  result.signal_name = kName;
  return result;
}

// ---------------------------------------------------------------------------
// Signal 4 — CLSD
// ---------------------------------------------------------------------------
SignalResult CrossAssetLiquidityStressDivergence::Compute(
    std::span<const double> bid_ask_spread,
    std::span<const double> next_ret) {
  const std::size_t n = bid_ask_spread.size();
  if (next_ret.size() != n || n < 4) {
    throw std::invalid_argument("CLSD: input size mismatch or n < 4");
  }

  double mad = MedianAbsDev(bid_ask_spread);
  mad = std::max(mad, kMinVariance);

  std::vector<double> sorted_ba(bid_ask_spread.begin(), bid_ask_spread.end());
  std::sort(sorted_ba.begin(), sorted_ba.end());
  double med = (n % 2 == 0)
      ? 0.5 * (sorted_ba[n / 2 - 1] + sorted_ba[n / 2])
      : sorted_ba[n / 2];

  std::vector<double> raw(n);
  for (std::size_t i = 0; i < n; ++i) {
    double liq_z = (bid_ask_spread[i] - med) / mad;
    // High bid-ask → liquidity stress → negative alpha (price dislocation).
    raw[i] = -std::copysign(std::sqrt(std::abs(liq_z)), liq_z);
  }

  auto z = CrossSectionalZScore(raw);
  auto rank = z;
  GaussianRankNormalize(rank);

  SignalResult result;
  result.raw_score = std::move(raw);
  result.z_score = std::move(z);
  result.rank_score = std::move(rank);
  result.ic = ComputeIC(result.raw_score, next_ret);
  result.icir = result.ic;
  result.signal_name = kName;
  return result;
}

// ---------------------------------------------------------------------------
// Signal 5 — CBPSM
// ---------------------------------------------------------------------------
SignalResult CentralBankPolicySurpriseMomentum::Compute(
    std::span<const double> ois_1y,
    std::span<const double> policy_rate,
    std::span<const double> ois_delta,
    std::span<const double> rolling_std,
    std::span<const double> regime_wt,
    std::span<const double> next_ret) {
  const std::size_t n = ois_1y.size();
  if (policy_rate.size() != n || ois_delta.size() != n ||
      rolling_std.size() != n || regime_wt.size() != n ||
      next_ret.size() != n || n < 4) {
    throw std::invalid_argument("CBPSM: input size mismatch or n < 4");
  }

  std::vector<double> raw(n);
  for (std::size_t i = 0; i < n; ++i) {
    double spread = ois_1y[i] - policy_rate[i];
    double normalized = (spread + ois_delta[i]) /
                        std::max(rolling_std[i], kMinVariance);
    raw[i] = normalized * std::clamp(regime_wt[i], 0.5, 1.0);
  }

  auto z = CrossSectionalZScore(raw);
  auto rank = z;
  GaussianRankNormalize(rank);

  SignalResult result;
  result.raw_score = std::move(raw);
  result.z_score = std::move(z);
  result.rank_score = std::move(rank);
  result.ic = ComputeIC(result.raw_score, next_ret);
  result.icir = result.ic;
  result.signal_name = kName;
  return result;
}

}  // namespace citadel::alpha
// ============================================================================
// HLS Branch — ISCF & MGD implementations
// ============================================================================

namespace citadel::alpha {

// Helpers (local to this TU)
namespace {

// Median of a span (in-place partial sort on copy).
double Median(std::span<const double> v) {
  std::vector<double> tmp(v.begin(), v.end());
  const std::size_t n = tmp.size();
  std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.end());
  if (n % 2 == 1) return tmp[n / 2];
  std::nth_element(tmp.begin(), tmp.begin() + n / 2 - 1, tmp.end());
  return 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

double Mad(std::span<const double> v, double med) {
  std::vector<double> abs_dev(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    abs_dev[i] = std::abs(v[i] - med);
  return Median(abs_dev);
}

// Rational approximation of the inverse error function (Winitzki 2003).
// Replaces Erfinv which is not in the C++ standard library.
double Erfinv(double x) {
  const double sgn = (x >= 0.0) ? 1.0 : -1.0;
  const double a = 0.147;
  const double ln1x2 = std::log(1.0 - x * x);
  const double term1 = 2.0 / (M_PI * a) + ln1x2 / 2.0;
  return sgn * std::sqrt(std::sqrt(term1 * term1 - ln1x2 / a) - term1);
}

} // namespace

// ----------------------------------------------------------------------------
// ISCF
// ----------------------------------------------------------------------------

SignalResult IdiosyncraticSupplyChainFlow::Compute(
    std::span<const double> spot,
    std::span<const double> deferred,
    std::span<const double> rvol,
    std::span<const double> macro_beta,
    std::span<const double> next_ret) {

  const std::size_t n = spot.size();
  if (n == 0 || n > kMaxAssets)
    throw std::invalid_argument("ISCF: n must be in (0, kMaxAssets]");

  constexpr double kEps = 1e-8;
  constexpr double kMaxZ = 4.0;

  // Step 1: volatility-normalised basis
  std::vector<double> basis(n);
  for (std::size_t i = 0; i < n; ++i)
    basis[i] = (spot[i] - deferred[i]) / std::max(rvol[i], kEps);

  // Step 2: robust z-score (median / MAD)
  double med = Median(basis);
  double mad = std::max(Mad(basis, med), kEps);

  std::vector<double> raw(n), z(n), rank(n);
  for (std::size_t i = 0; i < n; ++i) {
    double bz = std::clamp((basis[i] - med) / mad, -kMaxZ, kMaxZ);
    double beta = std::clamp(macro_beta[i], 0.0, 1.0);
    raw[i] = std::copysign(std::sqrt(std::abs(bz)), bz) * (1.0 - beta);
  }

  // Cross-sectional z-score
  double mu = std::accumulate(raw.begin(), raw.end(), 0.0) / static_cast<double>(n);
  double var = 0.0;
  for (auto x : raw) var += (x - mu) * (x - mu);
  double sigma = std::sqrt(var / static_cast<double>(n) + kMinVariance);
  for (std::size_t i = 0; i < n; ++i) z[i] = (raw[i] - mu) / sigma;

  // Gaussian rank normalisation
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b){
    return z[a] < z[b]; });
  for (std::size_t r = 0; r < n; ++r) {
    double u = (static_cast<double>(r) + 1.0) / (static_cast<double>(n) + 1.0);
    rank[order[r]] = std::sqrt(2.0) * Erfinv(2.0 * u - 1.0);
  }

  // IC
  double ic = 0.0;
  if (n > 1) {
    double sum_rx = 0, sum_ry = 0, sum_rxy = 0, sum_rx2 = 0, sum_ry2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum_rx += rank[i]; sum_ry += next_ret[i];
      sum_rxy += rank[i] * next_ret[i];
      sum_rx2 += rank[i] * rank[i];
      sum_ry2 += next_ret[i] * next_ret[i];
    }
    double dn = static_cast<double>(n);
    double denom = std::sqrt(std::max(
        (sum_rx2 - sum_rx*sum_rx/dn) * (sum_ry2 - sum_ry*sum_ry/dn), kMinVariance));
    ic = (sum_rxy - sum_rx*sum_ry/dn) / denom;
  }

  return SignalResult{
    .raw_score = raw, .z_score = z, .rank_score = rank,
    .ic = ic, .icir = 0.0, .signal_name = kName
  };
}

// ----------------------------------------------------------------------------
// MGD
// ----------------------------------------------------------------------------

SignalResult MacroGrowthDivergence::Compute(
    std::span<const double> pmi_surprise,
    std::span<const double> cpi_surprise,
    std::span<const double> emp_surprise,
    std::span<const double> fwd_expectation,
    std::span<const double> roll_std,
    std::span<const double> next_ret,
    double pmi_w, double cpi_w, double emp_w,
    double ema_alpha) {
  (void)ema_alpha;  // suppress unused-parameter warning

  const std::size_t n = pmi_surprise.size();
  if (n == 0 || n > kMaxAssets)
    throw std::invalid_argument("MGD: n must be in (0, kMaxAssets]");

  constexpr double kEps = 1e-8;

  std::vector<double> raw(n), z(n), rank(n);

  for (std::size_t i = 0; i < n; ++i) {
    double composite = pmi_w * pmi_surprise[i]
                     + cpi_w * cpi_surprise[i]
                     + emp_w * emp_surprise[i];
    double div = (composite - fwd_expectation[i]) / std::max(roll_std[i], kEps);
    raw[i] = div; // EMA smoothing applied across time in Python layer
  }

  double mu = std::accumulate(raw.begin(), raw.end(), 0.0) / static_cast<double>(n);
  double var = 0.0;
  for (auto x : raw) var += (x - mu) * (x - mu);
  double sigma = std::sqrt(var / static_cast<double>(n) + kMinVariance);
  for (std::size_t i = 0; i < n; ++i) z[i] = (raw[i] - mu) / sigma;

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b){
    return z[a] < z[b]; });
  for (std::size_t r = 0; r < n; ++r) {
    double u = (static_cast<double>(r) + 1.0) / (static_cast<double>(n) + 1.0);
    rank[order[r]] = std::sqrt(2.0) * Erfinv(2.0 * u - 1.0);
  }

  double ic = 0.0;
  if (n > 1) {
    double sum_rx = 0, sum_ry = 0, sum_rxy = 0, sum_rx2 = 0, sum_ry2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum_rx += rank[i]; sum_ry += next_ret[i];
      sum_rxy += rank[i] * next_ret[i];
      sum_rx2 += rank[i] * rank[i];
      sum_ry2 += next_ret[i] * next_ret[i];
    }
    double dn = static_cast<double>(n);
    double denom = std::sqrt(std::max(
        (sum_rx2 - sum_rx*sum_rx/dn) * (sum_ry2 - sum_ry*sum_ry/dn), kMinVariance));
    ic = (sum_rxy - sum_rx*sum_ry/dn) / denom;
  }

  return SignalResult{
    .raw_score = raw, .z_score = z, .rank_score = rank,
    .ic = ic, .icir = 0.0, .signal_name = kName
  };
}

} // namespace citadel::alpha
