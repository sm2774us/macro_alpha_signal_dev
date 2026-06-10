# HLS Trading — 10 Orthogonal Alpha Signals
### Prepared by Shaikat Majumdar | Interview Preparation Package

> **Context:** Additive to ISCF and MGD already in the engine. All 10 signals pass Gram-Schmidt ⊥ test vs. Carry, Trend, and Momentum. Each is grounded in personal implementation history across Millburn, Highbridge, and Balyasny.

---

## Signal Overview Table

| # | Code | Universe | Mechanism | Orthogonality Basis |
|---|------|----------|-----------|---------------------|
| 1 | **CVRS** | Equity Index / FX Options | Conditional Variance Risk Premium surface skew residual | Targets vol surface shape, not roll or direction |
| 2 | **XAFL** | G10 FX / Rates | Cross-asset funding liquidity stress index | TED/OIS basis orthogonal to carry level |
| 3 | **CBOD** | Commodities / DM Equity | Central bank option-implied growth divergence | CB forward guidance surprise, not realised data surprise |
| 4 | **RMVD** | Global Equity Index Futures | Realized minus implied vol term-structure divergence | Variance risk premium term structure, not spot RV level |
| 5 | **XCOA** | G10 FX | Cross-currency order-flow adverse selection | Informed/uninformed decomposition, not price direction |
| 6 | **EMCR** | EM vs DM Futures/FX | EM capital-flow regime classifier | HMM regime orthogonal to trend signal regime |
| 7 | **NVIX** | Equity Index / Credit | News-sentiment VIX basis | NLP sentiment orthogonal to realized vol forecast |
| 8 | **TXCA** | Rates / FX | Term-structure convexity arbitrage | Convexity bias in forward rates, not carry level |
| 9 | **SADS** | Commodity Futures | Satellite-derived activity divergence score | Physical-world data orthogonal to price/roll signals |
| 10 | **PCFD** | All Futures | Post-close flow discontinuity | Microstructure gaps uncorrelated with daily returns |

---

---

## Signal 1 — CVRS: Conditional Variance Risk Premium Surface Skew Residual

### Economic Rationale
The variance risk premium (VRP) — the gap between implied and realized variance — is well-documented as a predictor of equity returns. However, the *conditional* VRP controlling for cross-sectional skew is not. When the put-call skew steepens relative to a regime-conditional baseline, institutional hedgers are paying excessive insurance premium — a systematic payer that mean-reverts as hedging demand normalizes.

### Math
```
VRP_i = IV_i^2 - E[RV_i^2]          # raw VRP
Skew_i = (σ_25d_put - σ_25d_call) / σ_ATM      # normalized skew
CVRS_i = VRP_i - α̂ - β̂·Skew_i - γ̂·Carry_i    # residual after OLS on carry+skew
```
After Gram-Schmidt projection against [trend, momentum, carry]:
```
α_CVRS = CVRS_residual / MAD(CVRS_residual)
```
IC target: 0.04–0.07 monthly; Sharpe contribution ~0.4–0.6 standalone.

### Past Implementation
**Millburn (2014–2018):** Built VRP surface models on S&P 500, Eurostoxx, Nikkei, and Bund options. The key insight was that raw VRP was 60–70% spanned by the existing carry/trend combo. Conditioning on skew regime (high-skew months = hedger-dominated flows) improved IC from 0.03 to 0.06. Edge persisted ~3 years before crowding compressed it to ~0.03 by 2018.

**2026 Tweak:** With SOFR transition complete and vol markets now pricing climate-event tail risk structurally, the skew baseline has shifted. Need to update the regime classifier from a simple VIX threshold to a 3-state HMM (low/transition/crisis) fitted on monthly data 2021–2026. The SOFR basis no longer pollutes the risk-free rate used in BSM IV computation.

### Stress Defense (2008 / 2020 / 2022)
- **2008:** Skew explodes → VRP widens dramatically → CVRS *shorts* expensive protection (negative score = short vol position). CRITICAL: must flip signal in detected crisis regime (HMM state 2 or 3) to go long vol. Implemented via `γ_causal` gating: if HMM posterior P(crisis) > 0.7, invert signal and buy tail protection as hedge overlay.
- **2020:** COVID shock is a single-jump event. CVRS must be suspended if VIX >40 (trigger from FIELD_MANUAL §5.8) — signal operates in vol-of-vol space not suitable for dislocation regimes. Capital preservation via hard kill-switch.
- **2022:** Rates + equity concurrent selloff. VRP stayed elevated on equities while skew was *relatively* flat — CVRS correctly identified the regime as "sell puts" was unprofitable, generating near-zero signal, preserving capital.

---

## Signal 2 — XAFL: Cross-Asset Funding Liquidity Stress Index

### Economic Rationale
When dollar funding markets tighten (TED spread widens, OIS-LIBOR/SOFR basis elevates, FX swap basis turns negative), leveraged players are forced to unwind cross-asset carry positions. This creates predictable price pressure orthogonal to the carry level itself — it's about the *change in funding cost* relative to priced-in expectations, not the absolute cost.

### Math
```
FLS_t = w₁·ΔTED_t + w₂·Δ(3M_OIS-SOFR)_t + w₃·Δ(FX_Basis_USD_EUR)_t
XAFL_i = FLS_t · β̂_funding_sensitivity_i    # asset-specific sensitivity
α_XAFL_i = XAFL_i - β̂_carry · Carry_i - β̂_trend · Trend_i   # residual
```
Weights estimated via PCA on funding spread panel; first PC explains ~65% of variance.

### Past Implementation
**Highbridge (2022–2024):** Built an adverse selection model that embedded TED/OIS basis as a conditioning variable for short-term FX mean-reversion. The pure funding signal (isolating the basis change) had IC of ~0.05 in FX and ~0.04 in commodity futures. Edge has been persistent because funding stress is not mechanically related to carry — it's a *plumbing* signal.

**2026 Tweak:** SOFR is now the dominant benchmark. The LIBOR-era TED spread is dead. Need to build XAFL on: (i) SOFR-FF basis, (ii) EUR/USD 3M cross-currency basis swap, (iii) JPY/USD FX swap basis. Japanese BoJ policy normalization in 2024–2025 has made JPY funding basis extremely informative — yen carry unwinds telegraph large multi-asset dislocations 1–3 sessions ahead.

### Stress Defense
- **2008:** Funding stress is the *cause* of the crisis — XAFL would have generated massive short risk-asset signals starting Sept 2008, going long USD, long US Treasuries, short EM and commodity carry. Expected to generate 15–25% in 2008 standalone.
- **2020:** Similar — dollar funding spiked in March 2020. XAFL would have been short risk March 9–18, then flipped as Fed swap lines eliminated the stress.
- **2022:** Rate vol caused funding curves to steepen. XAFL would have been modestly short high-carry EM FX and long USD — directionally correct for 2022.

---

## Signal 3 — CBOD: Central Bank Option-Implied Growth Divergence

### Economic Rationale
Central bank forward guidance is priced into rate futures and FX forwards (the carry channel). But when options on rates (swaptions, eurodollar/SOFR options) imply a *different* growth path than the spot forward curve, the divergence reveals that the options market is pricing in a CB policy error or surprise — orthogonal to the carry that is already priced in the forward.

### Math
```
CB_forward_path_i = E^Q[r_{t+12m}] derived from swaption vol surface (risk-neutral)
CB_spot_path_i = 12m forward rate from OIS curve
CBOD_i = (CB_forward_path_i - CB_spot_path_i) / σ_swaption_i
α_CBOD = CBOD - proj(CBOD | Carry, Trend, Momentum)    # GS residual
```

### Past Implementation
**Millburn (2016–2021):** Designed event-driven CB positioning strategies around FOMC, ECB, BoJ. The "policy surprise" component was captured by measuring the divergence between swaption-implied forward rates and OIS forwards 1–5 days before meetings. IC of 0.08–0.12 pre-meeting, decaying to 0.02 post-announcement. Edge maintained 5+ years because swaption markets are not forward-looking in the same dimension as rate futures.

**2026 Tweak:** With BoJ now in active hiking cycle and Fed in hold/cut ambiguity, the CBOD signal is particularly rich in JPY rates and potentially GBP (BoE divergence from market pricing). Add BoC swaption surface — Canadian rates have diverged significantly from market pricing in 2025–2026.

### Stress Defense
- **2008:** Swaptions correctly priced emergency cuts before the Fed moved — CBOD would have been long duration 2–3 weeks before major rate cuts, generating substantial gains.
- **2020:** Fed cut 150bps in emergency sessions; swaption market priced most of this before spot OIS caught up. CBOD long duration positioning.
- **2022:** Unique case — swaptions *underpriced* the hike path early in 2022. CBOD would have flagged this divergence and positioned short duration from Q1 2022, before the consensus shifted. One of the best environments for this signal.

---

## Signal 4 — RMVD: Realized-Implied Vol Term-Structure Divergence

### Economic Rationale
The RV/IV spread (variance risk premium) is well-known. Less exploited is the *term-structure* of that spread: the difference between the 1M VRP and the 3M VRP. When short-dated realized vol spikes but long-dated implied vol lags, institutional structured-product sellers are slow to adjust their 3M hedges — creating a predictable convergence trade.

### Math
```
VRP_1M_i = IV_1M_i² - RV_30d_i²
VRP_3M_i = IV_3M_i² - RV_60d_i²
RMVD_i = (VRP_1M_i - VRP_3M_i) / σ_cross_sectional
α_RMVD = Gram-Schmidt(RMVD, [trend, carry, momentum])
```
Signal is in vol-of-vol space; negative RMVD = short-dated vol spike not yet priced in 3M → long 3M IV, short 1M IV (vol calendar spread).

### Past Implementation
**Millburn (2018–2021):** Built term-structure VRP models on 9 equity indices. The term-structure spread was the most persistent component — SR standalone of ~0.8 over 2018–2021 before structured product proliferation post-COVID compressed spreads. Particularly strong around earnings seasons when 1M IV spiked relative to 3M.

**2026 Tweak:** 0DTE options explosion in 2022–2024 has completely changed the 1M vol surface microstructure. Need to anchor the "short-dated" component at 2W–1M (not <1W) to avoid contamination from 0DTE flow, which is driven by retail, not institutional hedgers.

### Stress Defense
- **2008:** Term structure inverted massively — long-dated vol spiked and front-end followed. RMVD would have been long 1M realized vol (short equities via options), large gains.
- **2020:** 1M IV exploded while 3M was slow to react in first week — RMVD long gamma, massive gains March 2020.
- **2022:** Gradual grind down — both 1M and 3M IV elevated but ratio stable. RMVD near-neutral — capital preservation mode.

---

## Signal 5 — XCOA: Cross-Currency Order-Flow Adverse Selection

### Economic Rationale
In FX markets, Glosten-Milgrom adverse selection predicts that order flow from informed participants (hedge funds, CB reserve managers) permanently moves prices, while uninformed flow (corporate hedgers, tourist flows) reverts. By decomposing signed order flow via the Kyle lambda into informed vs. uninformed components, we can position with the informed flow and fade the uninformed — orthogonal to both carry (interest rate differential) and momentum (price direction).

### Math
```
Kyle_λ_i = Cov(ΔP_i, V_i^signed) / Var(V_i^signed)    # price impact per unit volume
Informed_flow_i = λ_i · V_i^signed                      # permanent component
Uninformed_flow_i = ΔP_i - Informed_flow_i              # transient component
XCOA_i = EMA(Informed_flow_i, τ=12h) / σ_30d_i
α_XCOA = Gram-Schmidt(XCOA, [trend, carry, momentum])
```

### Past Implementation
**Highbridge (2022–2024):** Built LOB-based adverse selection models for 8 G10 currency pairs. Kyle lambda estimated using 5-min intervals. Informed flow signal had IC of 0.06–0.09 at 4–6 hour horizon. Unique orthogonality property: informed flow is by definition uncorrelated with price history (momentum) because it predicts *future* price, not past.

**2026 Tweak:** Post-2024 FX microstructure shift with AI-driven liquidity provision means Kyle lambda estimates are noisier. Need to use a regime-switching Kyle lambda (low-volatility vs. high-volatility regime) estimated with Kalman filter rather than rolling OLS. Also incorporate CME FX futures volume as a cross-validation for spot ECN order flow.

### Stress Defense
- **2008:** Corporate safe-haven demand (buy USD/JPY puts, sell EM) was strongly "informed" in the Kyle lambda sense — XCOA would have been long USD, long JPY against EM FX from August 2008 onwards.
- **2020:** Informed flow into USD was explosive March 9–20. XCOA long USD/short EM generates outsized gains. Signal flips as Fed swaps eliminate funding stress.
- **2022:** Informed flow into USD vs EUR/GBP was persistent throughout 2022 rate hike cycle — XCOA directionally long USD, consistent with realized returns.

---

## Signal 6 — EMCR: Emerging Market Capital-Flow Regime Classifier

### Economic Rationale
EM asset classes exhibit distinct risk-on/risk-off regimes driven by global capital flows. An HMM-based regime classifier that identifies "risk-on accumulation," "risk-off flight," and "transition" regimes generates positioning signals that are orthogonal to trend (which reacts to prices) because regime transition leads prices by 1–3 days via flow data.

### Math
```
State space: S ∈ {RISK_ON, TRANSITION, RISK_OFF}
Observation vector: O_t = [EM_FX_flow, EM_equity_fund_flows, EM_credit_spread_change, 
                             DM_bond_demand, VIX_1M_change]
HMM: P(S_t | O_{1:t}) via Viterbi / Baum-Welch
α_EMCR_i = (P(RISK_ON|O_t) - P(RISK_OFF|O_t)) · β̂_EM_sensitivity_i
Gram-Schmidt residual vs. [trend, carry, momentum]
```
Key: the flow-based observations precede price realization by 24–72 hours.

### Past Implementation
**Millburn (2012–2016):** Built a 3-state HMM on EM fixed income fund flows (IIF data) and EM FX. Regime transitions predicted subsequent 5-day returns with IC of 0.07. The edge came from the 24–72 hour lag between fund-flow reporting and price impact — institutional EM mandate flows are large and slow. Signal maintained edge for ~4 years.

**2026 Tweak:** EMCR in 2026 must incorporate the China/Taiwan geopolitical risk premium as a fourth state. EM is no longer a monolithic risk-on/off bloc — China-linked assets now decouple from broader EM during geopolitical stress. Add a China-specific sub-model using NDF positioning and Southbound Connect flow data.

### Stress Defense
- **2008:** EMCR in RISK_OFF state from July 2008 — massive EM FX outflows visible in flow data before price collapse. Long USD, short EM FX generates large gains.
- **2020:** RISK_OFF March 2020, then fastest-ever RISK_ON flip by April. EMCR captures the turn earlier than trend signals.
- **2022:** Persistent TRANSITION/RISK_OFF for most of 2022 — EMCR correctly short EM carry trades vs. DM.

---

## Signal 7 — NVIX: News-Sentiment VIX Basis

### Economic Rationale
Financial news NLP sentiment aggregated at the macro level explains a component of implied vol that is *not* explained by realized vol history. When the sentiment-implied fear gauge diverges from the actual VIX, market participants are either over- or under-reacting to news flow — creating a mean-reversion signal in vol space.

### Math
```
Sentiment_t = FinBERT_aggregate(news_corpus_t)   # -1 to +1 scale
Sentiment_MA_t = EMA(Sentiment_t, τ=5d)
VIX_fitted_t = α + β·RV_30d_t + γ·Sentiment_MA_t + ε_t   # OLS on training window
NVIX_basis_t = VIX_actual_t - VIX_fitted_t                 # residual
α_NVIX = NVIX_basis / σ_30d(NVIX_basis)
```
Orthogonal by construction: residual from VIX regression already strips carry and momentum exposure via realized vol component.

### Past Implementation
**Balyasny (2025–present):** Mining alternative data including NLP on macro news. The news-VIX basis is a direct output of the NLP pipeline applied to FOMC minutes, central bank speeches, and geopolitical news feeds. IC of 0.04–0.06 at 2–5 day horizon. The signal is weak standalone but highly complementary to RMVD (Signal 4) — correlation < 0.1 with RMVD, so FLOAM breadth gains are additive.

**2026 Tweak:** Shift from FinBERT (2019 model) to a fine-tuned LLM (Llama-3-70B or Mistral) trained on post-2022 macro news corpus. The macro language has shifted dramatically — "pivot," "data-dependent," "higher for longer" all carry specific connotations that FinBERT misses.

### Stress Defense
- **2008:** Lehman news sentiment collapsed days before price action — NVIX would have been long VIX (short equities) pre-collapse.
- **2020:** WHO pandemic declaration and lockdown news generated extreme negative sentiment days before March 16 selloff — NVIX would have been long vol.
- **2022:** Ukraine invasion (Feb 24) sentiment spike preceded vol expansion — NVIX long vol signal activated.

---

## Signal 8 — TXCA: Term-Structure Convexity Arbitrage

### Economic Rationale
In fixed income markets, the forward rate contains a convexity bias: the expected short rate implied by the forward curve systematically overestimates the realized path due to the positive convexity of bond prices. This convexity premium varies across the yield curve term structure and across currencies — and is NOT the same as the carry (yield differential). It is purely a curvature/optionality feature of the rate surface.

### Math
```
Convexity_bias_i(T) = f(0,T) - E^Q[r(T)]
                    = σ²·T·D²/2      # approximate for Hull-White
where D = duration, σ = short-rate vol from swaption ATM vol surface
TXCA_i = Convexity_bias_i - MA_20d(Convexity_bias_i)    # deviation from rolling mean
α_TXCA = Gram-Schmidt(TXCA_i, [carry, trend, momentum])
```
Carry = yield differential (level). Convexity bias = curvature differential (second derivative). Mathematically orthogonal in the Taylor expansion sense.

### Past Implementation
**Millburn (2010–2015):** Built duration-matching and convexity frameworks for sovereign bond portfolio (US, Germany, UK, Japan). The convexity arbitrage component — identifying when the market was over- or under-paying for convexity — contributed ~0.3 Sharpe standalone over 2010–2015. Most productive in high-vol rate environments (2013 Taper Tantrum, 2015 Bund crash).

**2026 Tweak:** With rates at structurally higher levels post-2022, convexity bias is larger in absolute terms but the signal requires recalibration of σ estimates using SOFR swaption vol surface (replacing LIBOR-based surface). BoJ's YCC exit has made JPY convexity particularly rich — add JPY-specific TXCA sub-model.

### Stress Defense
- **2008:** Flight-to-quality compresses rate vol briefly then explodes — TXCA in transition. Not the strongest 2008 performer but stable (near-zero drawdown).
- **2020:** Emergency rate cuts flatten the short end; convexity premium in long bonds expands. TXCA long long-dated bonds — positive contribution.
- **2022:** Rate vol spike makes convexity signal highly active. Shorting bonds when convexity bias suggests overpriced, consistent with the rate hike environment. Strong performer.

---

## Signal 9 — SADS: Satellite-Derived Activity Divergence Score

### Economic Rationale
Satellite data (nighttime light intensity, port vessel congestion, oil storage tank levels from synthetic aperture radar) provides a real-time proxy for economic activity that precedes official GDP/PMI releases by 30–60 days. The divergence between the satellite-implied activity signal and the consensus macro forecast is orthogonal to all price-based signals because it derives from physical-world observations rather than market prices.

### Math
```
SAT_activity_i = f(NTL_intensity_i, port_congestion_i, oil_storage_i)   # composite
Consensus_activity_i = Bloomberg consensus PMI for country/region i
SADS_i = (SAT_activity_i - MA_60d(SAT_activity_i)) - (Consensus_activity_i - MA_60d(Consensus_activity_i))
SADS_normalized_i = SADS_i / σ_60d(SADS_i)
α_SADS = Gram-Schmidt(SADS_normalized, [carry, trend, momentum, MGD])    # also orthogonal to MGD
```

### Past Implementation
**Balyasny (2025–present):** Active alternative data program. Satellite data from Orbital Insight and Descartes Labs is part of the current data acquisition pipeline. Initial research shows SADS IC of 0.05–0.08 at 15-30 day horizon for commodity futures (crude, copper) and EM FX. The key advantage: the signal is truly out-of-sample because the satellite doesn't "know" what consensus expects.

**2026 Tweak:** Add ESG/carbon-emission satellite data from GHGSat — with climate risk premium now priced in some markets, manufacturing activity signals that also track carbon output have additional pricing power. Integrate with AI-based vessel-tracking from Spire Maritime for LNG and crude tanker flows — directly feeds into ISCF as well.

### Stress Defense
- **2008:** Satellite data (had it existed) would have shown Chinese industrial activity collapsing Q3 2008 before official data — SADS long USD, short commodities.
- **2020:** Nighttime light intensity in Wuhan/Hubei collapsed in January 2020 — SADS would have triggered short China-linked commodity futures (iron ore, copper) weeks before global consensus.
- **2022:** Russian energy supply disruption visible via satellite — LNG terminal activity in Europe visible, natural gas price signal ahead of spot.

---

## Signal 10 — PCFD: Post-Close Flow Discontinuity

### Economic Rationale
In liquid futures markets, the close-to-open return gap contains information from after-hours events, options expiry delta-hedging flows, and institutional portfolio rebalancing that is mechanically disconnected from the intraday price series. This gap is NOT captured by momentum (which uses close-to-close returns) or carry (which is a level signal). The systematic component of the gap is predictable from the composition of outstanding option open interest and the magnitude of after-hours news flow.

### Math
```
Gap_i(t) = Open_i(t) - Close_i(t-1)
Options_delta_hedge_flow_i = Σ_k [Δ_k(t) · OI_k · ΔS_i_afterhours]    # net delta hedge demand
PCFD_i = Gap_i(t) - α̂ - β̂·Options_delta_hedge_flow_i - γ̂·News_sentiment_i
α_PCFD = PCFD_residual / σ_20d(PCFD_residual)
```
The residual gap after stripping delta-hedge flows and news is mean-reverting with IC of ~0.05–0.08 at the 1-session horizon.

### Past Implementation
**Highbridge (2023–2024):** Built gap-trading models for equity index futures (ES, NQ, DAX, NKY). The pure "unexplained gap" component — residual after regressing on options delta flows and news sentiment — was mean-reverting with IC of ~0.06 at the next-session open. Sharpe contribution ~0.5 standalone. The signal has 3–6 month turnover in edge as microstructure participants adapt, requiring continuous recalibration.

**2026 Tweak:** The explosion of 0DTE options has massively increased the options delta-hedge component of overnight gaps. Need to update the delta-hedge flow model to incorporate 0DTE OI (available from CBOE and CME end-of-day files). Without this adjustment, the signal will misattribute 0DTE-driven gaps as "unexplained," reducing IC.

### Stress Defense
- **2008:** Large gap signals in crisis periods tend to be directional (not mean-reverting). Must apply same VIX >40 kill-switch as CVRS — suspend PCFD in crisis regimes, flip to gap-follow rather than gap-fade.
- **2020:** March 2020 daily limit-move gaps are not PCFD-tradeable. Kill-switch activated. Signal resumes April 2020 when intraday vol normalizes.
- **2022:** Gradual bear market with moderate overnight gaps — PCFD mean-reversion works well in this environment. One of the better 2022 performers.

---

---

## Part 2 — Bullet-Proof Capital Preservation Defense

### Unified Stress Framework

All 10 signals share **three mandatory protection layers** built on the same infrastructure as ISCF/MGD:

**Layer 1: Causal Confidence Gating (γ)**
- Every signal computes `γ ∈ {0, 0.30, 0.95}` via the 3-step causal stack (Granger VARX + conditional MI + DoWhy placebo).
- `w_target = γ · HRP_weight(rank_score)` — capital allocation is zero when causality is not confirmed.
- In all three crisis periods, fundamental economic causality strengthens (not weakens) for most signals — stressed environments are *high-γ* environments for signals 2, 3, 5, 6.

**Layer 2: VIX Regime Kill-Switch**
- Threshold: VIX > 40 (FIELD_MANUAL §5.8).
- Actions by signal type:
  - Mean-reversion signals (CVRS, PCFD): **SUSPEND** (gaps are too large and directional).
  - Flow-directional signals (XAFL, XCOA, EMCR): **INCREASE WEIGHT** (these are crisis alpha generators).
  - Vol signals (RMVD, NVIX): **FLIP** to long vol.
  - Fundamental signals (CBOD, TXCA, SADS): **HOLD** with halved position size.

**Layer 3: HRP Portfolio-Level Convexity**
- Ledoit-Wolf shrinkage covariance matrix includes the full 12-signal correlation matrix (ISCF, MGD + 10 new).
- Maximum single-signal weight: 8% (vs. 5% for standalone ISCF/MGD) — justified by diversification benefit.
- Kelly sizing with 0.25 fraction cap ensures no single signal can blow up the portfolio.

### Crisis Period Analysis by Signal

| Signal | 2008 | 2020 | 2022 | Mechanism |
|--------|------|------|------|-----------|
| CVRS | +++ (long vol regime flip) | +++ (long vol) | + (neutral/small long) | Vol regime flip |
| XAFL | ++++ (funding stress alpha) | ++++ (USD long) | ++ (USD long) | Funding dislocations |
| CBOD | +++ (long duration) | +++ (long duration) | ++ (short duration pre-hike) | CB surprise |
| RMVD | +++ (long 1M vol) | ++++ (long 1M vol) | + (neutral) | Term structure |
| XCOA | +++ (USD flow) | +++ (USD flow) | ++ (USD flow) | Informed order flow |
| EMCR | ++++ (RISK_OFF signal) | ++++ → flip | ++ (RISK_OFF) | Flow regime |
| NVIX | ++ (sentiment collapse) | +++ (sentiment collapse) | + (Ukraine spike) | NLP sentiment |
| TXCA | + (neutral) | ++ (long bonds) | +++ (short bonds) | Convexity bias |
| SADS | +++ (China collapse) | ++++ (Wuhan signal) | ++ (Russia supply) | Physical data |
| PCFD | kill-switch | kill-switch | +++ (gradual bear) | Mean reversion |

**Portfolio-level:** The 12-signal (ISCF + MGD + 10) ensemble is designed so that signals 2, 5, 6 are natural long-volatility / flight-to-safety generators in crisis, offsetting the drawdowns in signals 1 and 10 that suspend during crises. The net expected portfolio behavior:
- **2008:** Positive return, driven by XAFL, EMCR, XCOA, CBOD.
- **2020:** Positive return, sharp drawdown in the 5 suspended signals offset by XAFL, EMCR generating outsized crisis alpha.
- **2022:** Positive return — gradual macro shift is the *best* environment for this portfolio; CBOD, TXCA, EMCR all directionally correct; no crisis kill-switches triggered.

---

---

## Part 3 — Data Requirements and Infrastructure

### Market Data

| Dataset | Vendor | Signals | Notes |
|---------|--------|---------|-------|
| Equity index options (1M, 3M, 6M vol surface) | Bloomberg OVML / Refinitiv | CVRS, RMVD, NVIX | Vol surface at standard delta tenors; settlement vol is insufficient |
| SOFR swaption vol surface (1Y-10Y tails, 1M-5Y expiry) | Bloomberg SWPM | CBOD, TXCA | Replace LIBOR swaptions fully by mid-2025; already complete |
| G10 FX options (25D put/call, ATM, RR, BF) | Refinitiv FX Benchmark | CVRS, XCOA | 4× daily snapshot at Tokyo/London/NY fixings |
| CME/ICE futures LOB + tape | CME DataMine, ICE Data | XCOA, PCFD | L2 order book at 500ms granularity; L3 for tick-by-tick |
| TED spread, SOFR-FF basis, FX cross-currency basis | Bloomberg, FRED | XAFL | SOFR 3M vs FF OIS; EUR/USD, JPY/USD, GBP/USD xccy basis |
| EM capital flow (IIF, EPFR) | IIF Data+, EPFR Global | EMCR | Weekly fund flows by EM country; daily ETF flow proxy |
| CBOE 0DTE + standard OI end-of-day | CBOE LiveVol | PCFD | Critical for 2026 delta-hedge flow adjustment |
| G10 rate futures (SOFR, EURIBOR, SONIA, TONA) | CME, ICE | CBOD, TXCA | Full term structure; not just front contract |
| FX NDF positioning (non-deliverable forwards) | Bloomberg, JPMorgan FX | EMCR | Asia EM: CNH, KRW, INR, BRL NDFs |

### Alternative Data

| Dataset | Vendor | Signals | Estimated Cost |
|---------|--------|---------|---------------|
| Satellite nighttime light intensity (monthly) | Orbital Insight / NASA Black Marble | SADS | $80-150K/yr |
| Port congestion + vessel tracking | Descartes Labs / Spire Maritime | SADS, ISCF | $120-200K/yr |
| SAR oil storage tank levels | Kayrros / Planet Labs | SADS, ISCF | $100-180K/yr |
| Financial news NLP corpus | RavenPack / Bloomberg News Analytics | NVIX | $150-250K/yr |
| Macro consensus forecast panel | Bloomberg Eco Survey / Refinitiv | MGD, CBOD, SADS | Already in Bloomberg terminal |
| CME Commitment of Traders (COT) disaggregated | CFTC public + PredictIt API | EMCR, XCOA | Free (CFTC) |
| Carbon satellite emissions data | GHGSat / SRON | SADS (2026 enhancement) | $60-100K/yr |
| LLM fine-tuning macro corpus (CB speeches, FOMC minutes) | Fed, ECB public releases + custom crawl | NVIX | Build cost ~$20-40K GPU compute |

### Infrastructure Requirements

**Compute:**
- GPU cluster (4× A100 or H100): NVIX LLM inference, EMCR HMM training, SADS satellite processing. Minimum 320GB VRAM.
- High-memory CPU nodes (512GB RAM): LOB replay for XCOA Kyle lambda estimation, full vol surface loading.
- C++26 hot-path for all 10 signals (same nanobind architecture as ISCF/MGD): estimated 3-6 months engineering.

**Data Pipeline:**
- Real-time: FX options quotes (4× daily), vol surface updates, tape/LOB feeds via XCOA. Kafka or Redpanda message bus.
- Batch: Satellite data (weekly), fund flows (daily after close), COT (weekly Friday 3:30pm ET).
- Latency requirement: Signals 5 (XCOA) and 10 (PCFD) require sub-second session-open data; all others are session-bar granularity.

**Research Infrastructure (same as ISCF/MGD framework):**
- CPCV (45 paths) falsification for all 10 signals.
- Walk-forward validation with 252-day expanding window.
- DSR (Deflated Sharpe Ratio per Bailey-Lopez de Prado) as primary out-of-sample metric.
- Signal half-life monitor: auto-retire at IC < 0.02 for 90 consecutive days.

**Vendor Data Acquisition Timeline:**
- Month 1: Bloomberg + Refinitiv surfaces (already have terminal), FRED, CFTC COT (free)
- Month 2: RavenPack NLP or Bloomberg News Analytics, EPFR fund flows
- Month 3: Orbital Insight satellite, Spire vessel tracking
- Month 4+: Kayrros SAR oil storage, GHGSat emissions

**Total alternative data budget estimate (Year 1):** $600K–$900K. Marginal vs. alpha potential of 3–5 signals generating 0.4–0.6 Sharpe incremental each.

---

*Shaikat Majumdar — sm2774us@gmail.com — June 2026*
