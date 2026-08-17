// bindings.cpp — nanobind glue exposing C++26 hot-path to Python 3.13.
// Google C++ Style Guide.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "alpha_engine.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace ca = citadel::alpha;

// Helper: convert nanobind 1-D array to std::span<const double>.
template <typename T = double>
std::span<const T> ToSpan(
    const nb::ndarray<const T, nb::ndim<1>, nb::c_contig>& arr) {
  return {arr.data(), arr.size()};
}

// Helper: wrap SignalResult as a Python dict.
nb::dict ResultToDict(const ca::SignalResult& r) {
  nb::dict d;
  d["raw_score"] = r.raw_score;
  d["z_score"] = r.z_score;
  d["rank_score"] = r.rank_score;
  d["ic"] = r.ic;
  d["icir"] = r.icir;
  d["signal_name"] = std::string(r.signal_name);
  return d;
}

NB_MODULE(_citadel_alpha_cpp, m) {
  m.doc() = "Citadel Systematic Macro — C++26 Alpha Engine (nanobind)";

  // ------------------------------------------------------------------
  // Signal 1: GCSD
  // ------------------------------------------------------------------
  m.def(
      "compute_gcsd",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> ytm_10y,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> ytm_2y,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> next_ret) {
        auto res = ca::GlobalYieldCurveSlopeDivergence::Compute(
            ToSpan(ytm_10y), ToSpan(ytm_2y), ToSpan(next_ret));
        return ResultToDict(res);
      },
      "ytm_10y"_a, "ytm_2y"_a, "next_ret"_a,
      "Compute Global Yield Curve Slope Divergence signal.");

  // ------------------------------------------------------------------
  // Signal 2: RVIS
  // ------------------------------------------------------------------
  m.def(
      "compute_rvis",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> iv,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> rv,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> vrp_std,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> next_ret) {
        auto res = ca::RealisedImpliedVolSpread::Compute(
            ToSpan(iv), ToSpan(rv), ToSpan(vrp_std), ToSpan(next_ret));
        return ResultToDict(res);
      },
      "iv"_a, "rv"_a, "vrp_std"_a, "next_ret"_a,
      "Compute Realised vs. Implied Vol Spread Cross-Asset signal.");

  // ------------------------------------------------------------------
  // Signal 3: MSDI
  // ------------------------------------------------------------------
  m.def(
      "compute_msdi",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> surprises,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> ema_prev,
         double ema_alpha,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> rolling_std,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> next_ret) {
        auto res = ca::MacroSurpriseDiffusionIndex::Compute(
            ToSpan(surprises), ToSpan(ema_prev), ema_alpha,
            ToSpan(rolling_std), ToSpan(next_ret));
        return ResultToDict(res);
      },
      "surprises"_a, "ema_prev"_a, "ema_alpha"_a, "rolling_std"_a,
      "next_ret"_a, "Compute Macro Surprise Diffusion Index signal.");

  // ------------------------------------------------------------------
  // Signal 4: CLSD
  // ------------------------------------------------------------------
  m.def(
      "compute_clsd",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> bid_ask,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> next_ret) {
        auto res = ca::CrossAssetLiquidityStressDivergence::Compute(
            ToSpan(bid_ask), ToSpan(next_ret));
        return ResultToDict(res);
      },
      "bid_ask"_a, "next_ret"_a,
      "Compute Cross-Asset Liquidity Stress Divergence signal.");

  // ------------------------------------------------------------------
  // Signal 5: CBPSM
  // ------------------------------------------------------------------
  m.def(
      "compute_cbpsm",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> ois_1y,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> policy_rate,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> ois_delta,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> rolling_std,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> regime_wt,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> next_ret) {
        auto res = ca::CentralBankPolicySurpriseMomentum::Compute(
            ToSpan(ois_1y), ToSpan(policy_rate), ToSpan(ois_delta),
            ToSpan(rolling_std), ToSpan(regime_wt), ToSpan(next_ret));
        return ResultToDict(res);
      },
      "ois_1y"_a, "policy_rate"_a, "ois_delta"_a, "rolling_std"_a,
      "regime_wt"_a, "next_ret"_a,
      "Compute Central Bank Policy Surprise Momentum signal.");

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------
  m.def(
      "ledoit_wolf_shrink",
      [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> cov_flat,
         std::size_t n) {
        ca::LedoitWolfShrink({cov_flat.data(), cov_flat.size()}, n);
      },
      "cov_flat"_a, "n"_a,
      "Apply Ledoit-Wolf shrinkage in-place on flattened NxN covariance.");

  m.def(
      "gaussian_rank_normalize",
      [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> values) {
        ca::GaussianRankNormalize({values.data(), values.size()});
      },
      "values"_a, "Gaussian rank-normalize array in-place (Blom formula).");

  m.def(
      "compute_ic",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> signal,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> returns) {
        return ca::ComputeIC(ToSpan(signal), ToSpan(returns));
      },
      "signal"_a, "returns"_a, "Compute Spearman rank IC.");

  m.def("deflated_sharpe_ratio", &ca::DeflatedSharpeRatio,
        "sr"_a, "sr_benchmark"_a, "T"_a, "skew"_a, "kurt"_a, "N_trials"_a,
        "Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).");
  // ------------------------------------------------------------------
  // Proprietary Trading Firm — Signal 6: ISCF
  // ------------------------------------------------------------------
  m.def("compute_iscf",
    [](const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& spot,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& deferred,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& rvol,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& macro_beta,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& next_ret) {
      return ResultToDict(ca::IdiosyncraticSupplyChainFlow::Compute(
          ToSpan(spot), ToSpan(deferred), ToSpan(rvol),
          ToSpan(macro_beta), ToSpan(next_ret)));
    },
    "spot"_a, "deferred"_a, "rvol"_a, "macro_beta"_a, "next_ret"_a,
    "ISCF hot-path: volatility-normalised backwardation z-score.");

  // ------------------------------------------------------------------
  // Proprietary Trading Firm — Signal 7: MGD
  // ------------------------------------------------------------------
  m.def("compute_mgd",
    [](const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& pmi,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& cpi,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& emp,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& fwd_exp,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& roll_std,
       const nb::ndarray<const double, nb::ndim<1>, nb::c_contig>& next_ret,
       double pmi_w, double cpi_w, double emp_w, double ema_alpha) {
      return ResultToDict(ca::MacroGrowthDivergence::Compute(
          ToSpan(pmi), ToSpan(cpi), ToSpan(emp),
          ToSpan(fwd_exp), ToSpan(roll_std), ToSpan(next_ret),
          pmi_w, cpi_w, emp_w, ema_alpha));
    },
    "pmi"_a, "cpi"_a, "emp"_a, "fwd_exp"_a, "roll_std"_a, "next_ret"_a,
    "pmi_w"_a = 0.40, "cpi_w"_a = 0.30, "emp_w"_a = 0.30, "ema_alpha"_a = 0.333,
    "MGD hot-path: composite macro surprise vs forward-priced expectation.");
}
