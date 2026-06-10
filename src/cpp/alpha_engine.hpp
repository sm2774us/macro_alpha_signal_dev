// Copyright 2025 Citadel Systematic Macro Pod
// Licensed under Apache 2.0
//
// alpha_engine.hpp — C++26 hot-path signal computation engine.
// Five orthogonal alpha signals, exposed via nanobind to Python 3.13.
// Follows Google C++ Style Guide.
// All hot-path structs follow data-oriented design (SoA) for cache efficiency.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace citadel::alpha {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr double kAnnualizationFactor = std::sqrt(252.0);
inline constexpr double kMinVariance = 1e-12;
inline constexpr std::size_t kMaxAssets = 64;

// ---------------------------------------------------------------------------
// Signal result — per-asset, per-signal output (SoA layout)
// ---------------------------------------------------------------------------
struct SignalResult {
  std::vector<double> raw_score;    // unscaled signal [-inf, +inf]
  std::vector<double> z_score;      // cross-sectionally z-scored signal
  std::vector<double> rank_score;   // Gaussian-rank normalized [-1, 1]
  double ic;                        // information coefficient vs. next_ret
  double icir;                      // IC / rolling_std(IC)
  std::string_view signal_name;
};

// ---------------------------------------------------------------------------
// Signal 1 — Global Yield Curve Slope Divergence (GCSD)
//
// Thesis: Cross-country divergence in yield curve steepness (10Y-2Y spread)
// predicts currency and rate futures direction, orthogonal to carry (which
// uses level differences) and trend (which uses price momentum).
//
// α_GCSD_i = [slope_i - median(slope)] / MAD(slope)
//            where slope = log(ytm_10y) - log(ytm_2y)
// ---------------------------------------------------------------------------
class GlobalYieldCurveSlopeDivergence {
 public:
  static constexpr std::string_view kName = "GCSD";

  // Compute raw signal scores across N assets.
  // ytm_10y, ytm_2y: yield vectors of length N (in decimal, e.g. 0.045)
  // next_ret:        forward 1-period returns for IC computation (length N)
  static SignalResult Compute(std::span<const double> ytm_10y,
                              std::span<const double> ytm_2y,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// Signal 2 — Realised vs. Implied Volatility Spread Cross-Asset (RVIS)
//
// Thesis: The variance risk premium (VRP = IV - RV) predicts next-period
// returns across equities, FX, and rates. Orthogonal to carry (yield-based)
// and trend (price-based). VRP is mean-reverting and economically grounded.
//
// α_RVIS_i = (IV_i - RV_i) / σ_rolling(VRP_i, T=60)
// ---------------------------------------------------------------------------
class RealisedImpliedVolSpread {
 public:
  static constexpr std::string_view kName = "RVIS";

  // iv:     implied vol series (rolling window, shape NxT)
  // rv:     realised vol series  (rolling window, shape NxT)
  // window: lookback for VRP z-scoring
  static SignalResult Compute(std::span<const double> iv_current,
                              std::span<const double> rv_current,
                              std::span<const double> vrp_rolling_std,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// Signal 3 — Macro Surprise Diffusion Index (MSDI)
//
// Thesis: Economic data surprises (actual - consensus) exhibit short-term
// autocorrelation across countries. A cross-country diffusion index of
// normalised surprises predicts subsequent asset returns at 1-5 week horizons.
// Orthogonal to trend (price, not data) and carry (yield levels, not flow).
//
// α_MSDI_i = EMA(surprise_i, τ=10) / rolling_std(surprise_i, T=120)
// surprise_i = (actual_i - consensus_i) / σ_historical_i
// ---------------------------------------------------------------------------
class MacroSurpriseDiffusionIndex {
 public:
  static constexpr std::string_view kName = "MSDI";

  // surprises: normalised surprise scores per asset, length N
  // ema_alpha: EMA decay parameter (e.g., 2/(10+1))
  static SignalResult Compute(std::span<const double> surprises,
                              std::span<const double> ema_prev,
                              double ema_alpha,
                              std::span<const double> rolling_std,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// Signal 4 — Cross-Asset Liquidity Stress Divergence (CLSD)
//
// Thesis: Bid-ask spreads and repo rates embed real-time funding stress.
// Divergence between asset-level liquidity stress and cross-asset median
// predicts near-term price dislocation and reversal. Orthogonal to all
// price-based signals.
//
// α_CLSD_i = -sign(liq_z_i) * |liq_z_i|^0.5
// liq_z_i  = (bid_ask_i - median) / MAD   (robust z-score)
// ---------------------------------------------------------------------------
class CrossAssetLiquidityStressDivergence {
 public:
  static constexpr std::string_view kName = "CLSD";

  // bid_ask_spread: per asset, length N
  // next_ret: forward returns for IC
  static SignalResult Compute(std::span<const double> bid_ask_spread,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// Signal 5 — Central Bank Policy Surprise Momentum (CBPSM)
//
// Thesis: Unexpected shifts in central bank forward guidance (OIS vs. policy
// rate consensus) exhibit multi-week momentum in rate and FX markets.
// Distinct from carry (static yield differential) and trend (price history).
//
// α_CBPSM_i = Δ(ois_1y_i - policy_rate_i) / σ_rolling * regime_weight_i
// regime_weight: 1.0 in expansion, 0.5 in contraction (HMM-derived)
// ---------------------------------------------------------------------------
class CentralBankPolicySurpriseMomentum {
 public:
  static constexpr std::string_view kName = "CBPSM";

  // ois_1y:       1-year OIS rate per country, length N
  // policy_rate:  official policy rate per country, length N
  // ois_delta:    change in OIS over past week, length N
  // rolling_std:  historical std of (ois - policy) delta, length N
  // regime_wt:    HMM-derived regime weight [0.5, 1.0], length N
  static SignalResult Compute(std::span<const double> ois_1y,
                              std::span<const double> policy_rate,
                              std::span<const double> ois_delta,
                              std::span<const double> rolling_std,
                              std::span<const double> regime_wt,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// Signal Combination — HRP-weighted composite
// ---------------------------------------------------------------------------
struct CompositeResult {
  std::vector<double> weights;          // N asset weights
  std::vector<double> composite_score;  // combined signal per asset
  double portfolio_sharpe;
  double portfolio_ir;
  std::array<double, 5> signal_ic;      // per-signal IC
  std::array<double, 5> signal_icir;    // per-signal ICIR
};

// Ledoit-Wolf shrinkage on a sample covariance matrix (NxN, row-major).
// Returns shrunk covariance in-place.
void LedoitWolfShrink(std::span<double> cov_matrix, std::size_t n);

// Compute Gaussian rank normalization across a vector (modifies in-place).
// Maps ranks to N(0,1) quantiles via Blom formula.
void GaussianRankNormalize(std::span<double> values);

// Compute Information Coefficient (Spearman rank correlation).
double ComputeIC(std::span<const double> signal,
                 std::span<const double> returns);

// Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
// sr: estimated Sharpe, sr_benchmark: max Sharpe from N_trials,
// T: number of observations, skew: return skewness, kurt: excess kurtosis.
double DeflatedSharpeRatio(double sr, double sr_benchmark, std::size_t T,
                            double skew, double kurt, std::size_t N_trials);

// Compute robust median absolute deviation.
double MedianAbsDev(std::span<const double> x);

// Compute EMA for a scalar update.
inline double EMAUpdate(double prev, double new_val, double alpha) noexcept {
  return alpha * new_val + (1.0 - alpha) * prev;
}

// ---------------------------------------------------------------------------
// HLS Branch — Signal 6: ISCF (Idiosyncratic Supply Chain Flow)
//
// α_ISCF_i = sign(z_i) * sqrt(|z_i|) * (1 - β_macro_i)
// where z_i = (basis_i - median(basis)) / MAD(basis)
//       basis_i = (spot_i - deferred_i) / max(rvol_i, ε)
// Gram-Schmidt residualisation against trend/momentum/carry applied in Python.
// ---------------------------------------------------------------------------
class IdiosyncraticSupplyChainFlow {
 public:
  static constexpr std::string_view kName = "ISCF";

  static SignalResult Compute(std::span<const double> spot,
                              std::span<const double> deferred,
                              std::span<const double> rvol,
                              std::span<const double> macro_beta,
                              std::span<const double> next_ret);
};

// ---------------------------------------------------------------------------
// HLS Branch — Signal 7: MGD (Real-Time Macro Growth Divergence)
//
// S_i = w_PMI*PMI_i + w_CPI*CPI_i + w_EMP*EMP_i
// α_MGD_i = EMA_smooth[(S_i - FWD_i) / max(σ_roll_i, ε)]
// ---------------------------------------------------------------------------
class MacroGrowthDivergence {
 public:
  static constexpr std::string_view kName = "MGD";

  static SignalResult Compute(std::span<const double> pmi_surprise,
                              std::span<const double> cpi_surprise,
                              std::span<const double> emp_surprise,
                              std::span<const double> fwd_expectation,
                              std::span<const double> roll_std,
                              std::span<const double> next_ret,
                              double pmi_w = 0.40,
                              double cpi_w = 0.30,
                              double emp_w = 0.30,
                              double ema_alpha = 0.333);

}; // class MacroGrowthDivergence

}  // namespace citadel::alpha
