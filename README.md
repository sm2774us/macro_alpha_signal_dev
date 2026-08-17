# Proprietary Trading Firm — Alpha Engine — ISCF & MGD Systematic Macro Signals

> **Two orthogonal, causally-validated alpha signals** targeting unspanned dimensions of the systematic macro return space at mid-to-high frequency (4× daily rebalancing).  
> Python 3.13 · C++26 · nanobind · uv · CMake · yfinance/FRED · Plugin data architecture · Docker · GitHub Actions

---
---

[![CI — Proprietary Trading Firm Alpha Engine (ISCF + MGD + Causal Stack)](https://github.com/sm2774us/macro_alpha_signal_dev/actions/workflows/ci.yml/badge.svg)](https://github.com/sm2774us/macro_alpha_signal_dev/actions/workflows/ci.yml)

---
---

## 📑 Table of Contents

- [Synopsis](#synopsis)
- [Solution Architecture](#solution-architecture)
- [Signal Explanation and Economic Rationale](#signal-explanation-and-economic-rationale)
- [Mathematical Background — From First Principles](#mathematical-background--from-first-principles)
- [ML and Statistical Methodology](#ml-and-statistical-methodology)
- [Data Provider Architecture](#data-provider-architecture)
- [Causal Validation Framework](#causal-validation-framework)
- [4× Daily Rebalancing Engine](#4-daily-rebalancing-engine)
- [Signal Health Monitoring and Retirement](#signal-health-monitoring-and-retirement)
- [Six-Month Deployment Plan](#six-month-deployment-plan)
- [Notebook Research and Plots](#notebook-research-and-plots)
- [Build Compile and Run Instructions](#build-compile-and-run-instructions)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Dissertation](#dissertation)

---
---

## Synopsis

[🔝 Back to Top](#-table-of-contents)

Proprietary Trading Firm operates systematic macro across FX, commodities, futures, and rates at mid-to-high frequency — rebalancing approximately four times per day. The research culture is **statistics-first**: rigorous thinking over technique, real predictive content separated from noise, linear and non-linear models deployed only where theoretically justified.

This engine implements two signals that are structurally orthogonal to trend, momentum, and carry — maximising FLOAM breadth $\text{IR} = \text{IC}\sqrt{N_{\text{breadth}}}$ without cannibalising existing edge:

| Signal | Code | Universe | Frequency | Core Mechanism |
|--------|------|----------|-----------|----------------|
| Idiosyncratic Supply Chain Flow | **ISCF** | Metals/Energy Futures | 4× daily | Physical delivery-premium backwardation orthogonal to roll yield |
| Real-Time Macro Growth Divergence | **MGD** | G10 FX Forward Panel | 4× daily | Nowcast composite surprise vs. forward-priced growth expectation |

Each signal passes a **3-step causal validation stack** and receives a causal confidence factor $\gamma \in \{0, 0.30, 0.95\}$ that directly gates capital allocation and rebalancing decisions per session.

---
---

## Solution Architecture

[🔝 Back to Top](#-table-of-contents)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     PROPRIETARY TRADING FIRM — ALPHA ENGINE — SYSTEM ARCHITECTURE   │
└─────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────── Data Provider Layer (Design-by-Contract) ────────────────────┐
  │                                                                              │
  │  AbstractISCFProvider ──► YFinanceISCFProvider  (free: ETF proxies)          │
  │                      └──► HLSISCFProvider       (Day-1: LME/CME feeds)       │
  │  AbstractMGDProvider  ──► YFinanceMGDProvider   (free: FRED+yfinance)        │
  │                      └──► HLSMGDProvider        (Day-1: Bloomberg PMI)       │
  │  get_iscf_provider(mode="yfinance"|"proprietary") ◄── single factory entry   │
  │  ISCFMarketData / MGDMarketData ◄── frozen dataclass contracts               │
  └──────────────────────────┬───────────────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────── Signal Computation (Python + C++26) ───────────────────┐
  │                                                                        │
  │  signals_hls.py                                                        │
  │  ├─ compute_iscf()  basis/vol robust z-score → √|z|·sign·(1-β_macro)   │
  │  │                  → Gram-Schmidt ⊥ [trend, momentum, carry]          │
  │  └─ compute_mgd()   composite(PMI,CPI,EMP) − EMA[composite] / σ_roll   │
  │                     → EMA(τ=5) smoothing → Gram-Schmidt ⊥              │
  │                                                                        │
  │  C++26 hot-path (_citadel_alpha_cpp.so via nanobind):                  │
  │  ├─ IdiosyncraticSupplyChainFlow::Compute()  O(N log N) median/MAD     │
  │  └─ MacroGrowthDivergence::Compute()  vectorised weighted sum          │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             │ SignalResult: rank_score, IC, ICIR
                             ▼
  ┌──────────────── Causal Validation (3-step) ────────────────────────────┐
  │                                                                        │
  │  Step 1: granger_causality_varx()   VARX + HAC Newey-West (L=12)       │
  │  Step 2: conditional_independence_test()  CMI proxy (partial ρ)        │
  │  Step 3: dowhy_refutation()  Placebo + MBB Policy Invariance           │
  │  → γ ∈ {0.00, 0.30, 0.95}  → REJECT / BETA_PROXY / PASS                │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             │ CausalStackResult + γ
                             ▼
  ┌──────────────── Rebalancing Engine (4× daily) ─────────────────────────┐
  │                                                                        │
  │  rebalance.py                                                          │
  │  ├─ Session: ASIA | LONDON | NY_OPEN | NY_CLOSE                        │
  │  ├─ w_target = γ · HRP_weights(rank_scores)                            │
  │  ├─ TCA: TC(bps) = (bid-ask/2)·turnover + γ_impact·√(turnover/ADV)     │
  │  └─ Go/No-Go: γ < 0.25 → suspend; turnover > 50% → split sessions      │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             │ RebalanceDecision per session
                             ▼
  ┌──────────────── Falsification + Portfolio ─────────────────────────────┐
  │                                                                        │
  │  falsification.py:  CPCV (45 paths), DSR (Bailey-LdP), BH/Bonferroni   │
  │  portfolio.py:      HRP (Ledoit-Wolf), Kelly sizing                    │
  │  constants.py:      All empirical constants §1–§5.8                    │
  └──────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────── CLI + CI/CD ───────────────────────────────────────────┐
  │                                                                        │
  │  hls-alpha hls-run     → synthetic pipeline                            │
  │  hls-alpha hls-live    → real market data (yfinance/FRED or Proprietary Trading Firm)
  │  hls-alpha hls-monitor → KPI + half-life + rebalancing (cron 4×/day)   │
  │  hls-alpha benchmark   → C++ vs Python timing                          │
  │                                                                        │
  │  GitHub Actions (4× daily cron: 07:00 / 12:00 / 17:00 / 21:00 UTC):    │
  │  ├─ python-build-test  → pytest + coverage                             │
  │  ├─ cpp-build-test     → CMake/Ninja + GTest                           │
  │  ├─ signal-health-monitor → hls-alpha hls-monitor → KPI_REPORT.md      │
  │  │   └─ $GITHUB_STEP_SUMMARY (inline PR/run report)                    │
  │  ├─ benchmark          → timing artifact                               │
  │  └─ docker-build       → multi-stage smoke test                        │
  └────────────────────────────────────────────────────────────────────────┘
```

---
---

## Signal Explanation and Economic Rationale

[🔝 Back to Top](#-table-of-contents)

### ISCF — Idiosyncratic Supply Chain Flow

**Why it works:** Commodity futures markets embed physical delivery constraints that are structurally invisible to pure price-momentum or yield-carry signals. When warehouse inventory falls below critical thresholds — LME copper stocks below 50,000 MT, WTI crude at Cushing below seasonal norms — the forward curve enters steep backwardation as physical market participants pay a premium for immediate delivery. This premium reverts as inventory normalises, generating a predictable mean-reversion return.

**Orthogonality argument:** Carry signals exploit $f(\text{roll yield})$. Roll yield = $(F_{t+1} - F_t)/F_t$ captures the average forward curve slope. ISCF exploits $f\!\left(\frac{\text{spot} - \text{deferred}}{\sigma^{rv}}\right)$ — the *basis-to-vol ratio*, a function of physical delivery premium net of volatility. After Gram-Schmidt projection $\hat{\alpha} \perp [\text{trend}, \text{mom}, \text{carry}]$, pairwise $R^2 < 0.15$.

**Macro-beta suppression:** Assets with high macro-beta (correlated with global risk-on/off) have their signal weight suppressed by $(1 - \beta^{\text{macro}})$. This isolates the supply-chain component from the broad commodity-as-risk-asset behaviour captured by existing signals.

### MGD — Real-Time Macro Growth Divergence

**Why it works:** FX forward curves price in expected interest rate differentials, which embed growth expectations. When realised economic releases (PMI, CPI, NFP) surprise relative to consensus, spot FX rates reprice — but not instantaneously. Institutional flow lags (portfolio rebalancing, options delta hedges) delay full price adjustment by 1–3 periods. MGD captures this window by measuring the divergence between the composite nowcast surprise and the EMA-smoothed forward-curve expectation.

**Orthogonality argument:** Carry = $f(r - r^*)$, current yield differential. MGD = $f\!\left(\frac{S - \mathbb{E}^{\text{FWD}}[S]}{\sigma}\right)$ where $S$ = weighted macro surprise. Since $S$ is a residual from consensus — by construction orthogonal to priced-in rate levels — $\text{Cov}(\alpha^{\text{MGD}}, \alpha^{\text{carry}}) \to 0$ in expectation.

**Mid-to-high frequency fit:** At 4× daily rebalancing, intraday data releases (flash PMI, inflation prints, NFP) arrive within the holding period. MGD is specifically designed to capture the lag between release and full price adjustment — the signal has highest predictive content in the 1–6 hour window post-release.

> [!IMPORTANT]
> **Nowcasting** is the prediction of the current or very near-future state of the economy (like current-quarter GDP) using real-time, high-frequency data, because official metrics often have long publication lags. It helps policymakers and investors make decisions without waiting for finalized reports. [1, 2, 3]  
>
> **Nowcast Composite Surprise**
>
> • Measures how much incoming economic data exceeds or falls short of real-time predictions. 
> • Positive surprise: Data is stronger than the nowcast. 
> • Negative surprise: Data is weaker than the nowcast.
>
> **Forward-Priced Growth Expectation**
>
> • Represents the economic growth already priced into financial markets (e.g., bond yields, equity futures). 
> • Acts as a baseline market expectation for future periods. 
> • Above-expectation pricing: Anticipates accelerated growth. 
> • Below-expectation pricing: Anticipates a slowdown.
>
> **The Dynamic Between the Two**
>
> • Market Reaction: Markets react when the nowcast composite surprise clashes with forward-priced growth expectations. 
> • Positive Shock: Positive nowcast surprises cause markets to reprice higher if forward expectations were too low. 
> • Negative Shock: Negative nowcast surprises trigger sell-offs if forward pricing was overly optimistic.

---
---

## Mathematical Background — From First Principles

[🔝 Back to Top](#-table-of-contents)

### 1. Fundamental Law of Active Management (FLOAM)

From Grinold (1989), the Information Ratio of an active strategy is:

$$\text{IR} = \text{IC} \cdot \sqrt{\text{Breadth}}$$

where $\text{IC} = \mathbb{E}\!\left[\text{corr}(\hat{r}_i, r_i)\right]$ and Breadth = number of independent forecasts per period. Adding $K$ orthogonal signals satisfying $\langle\alpha^k, \alpha^j\rangle = 0\ \forall k \neq j$ multiplies breadth:

$$\text{Breadth}_{\text{total}} = K \cdot B_{\text{single}}$$

yielding $\sqrt{K}$ IR improvement with no additional IC requirement.

**Empirical calibration for HLS (mid-to-high frequency):**  
$\text{IC} \in [0.02, 0.06]$ for systematic macro at daily resolution.  
With 4× daily rebalancing: $\text{Breadth}_{\text{session}} = 4 \cdot 252 = 1008$ per year.  
$\text{IR} = 0.04 \times \sqrt{1008} \approx 1.27$ from a single signal.  
Two orthogonal signals: $\text{IR} = 0.04 \times \sqrt{2016} \approx 1.80$.

### 2. ISCF Signal: Derivation from Physical Market Microstructure

**Step 1 — Convenience yield and physical basis:**  
Under no-arbitrage for storable commodities (Working 1949):
$$F(t, T) = S(t) \cdot e^{(r + u - c)(T-t)}$$
where $u$ = storage cost, $c$ = convenience yield. The *convenience yield* $c$ rises sharply when inventory falls below critical levels (Fama & French 1988), creating backwardation: $F < S$.

**Step 2 — Volatility-normalised basis:**
$$b_i = \frac{S_i - F_i^{\text{deferred}}}{\max(\sigma_i^{rv}, \varepsilon)}, \quad \sigma_i^{rv} = \sqrt{\frac{252}{20}\sum_{\tau=1}^{20}(\log S_{i,t-\tau+1} - \log S_{i,t-\tau})^2}$$

Dividing by realised vol converts basis from dollar-units to a standardised *excess-of-risk* measure, making it comparable across Copper, WTI, and Gold simultaneously.

**Step 3 — Robust cross-sectional z-score (MAD normalisation):**

Classical z-scores $(x - \bar{x})/\sigma$ are sensitive to outliers from flash crashes and commodity squeezes. We use:

$$z_i = \mathop{\mathrm{clip}}\!\left(\frac{b_i - \mathop{\mathrm{median}}\limits_j(b_j)}{\mathop{\mathrm{MAD}}\limits_j(b_j) + \varepsilon}, \; -z_{\max}, z_{\max}\right)$$

$\mathop{\mathrm{MAD}} = \mathop{\mathrm{median}}\limits_j|b_j - \mathop{\mathrm{median}}(b)|$
 has breakdown point 0.5 vs. 0 for the mean, making it resistant to up to 50% contaminated observations.

**Step 4 — Square-root dampening and macro-beta suppression:**

$$\alpha_i^{\text{ISCF,raw}} = \mathop{\mathrm{sign}}(z_i) \cdot |z_i|^{1/2} \cdot (1 - \beta_i^{\text{macro}})$$


The $\sqrt{|\cdot|}$ transformation achieves two goals: (1) reduces the fat-tail kurtosis of the raw z-score distribution, and (2) applies diminishing returns — very steep backwardation (large $|z|$) receives less marginal weight than a linear signal would assign, improving Sharpe stability.

**Step 5 — Gram-Schmidt orthogonalisation:**
$$\hat{\alpha}^{\text{ISCF}} = \alpha^{\text{ISCF,raw}} - \sum_{k \in \{\text{trend, mom, carry}\}} \frac{\langle\alpha^{\text{ISCF,raw}}, \mathbf{f}^k\rangle}{\langle\mathbf{f}^k, \mathbf{f}^k\rangle} \mathbf{f}^k$$

By construction: $\langle\hat{\alpha}^{\text{ISCF}}, \mathbf{f}^k\rangle = 0\ \forall k$. VIF < 5.0 and pairwise $R^2 < 0.15$ are enforced as hard gates.

### 3. MGD Signal: Derivation from Rational Expectations Theory

**Step 1 — Efficient markets and the surprise residual:**  
Under rational expectations (Muth 1961), priced-in expectations are unbiased: $\mathbb{E}[S_{t+1}|I_t] = F_{t+1}^{\text{FWD}}$ where $F^{\text{FWD}}$ is the forward-curve-implied expectation. The *economic surprise* is:

$$
\Delta_{i,t} = \frac{\text{actual}_{i,t} - \text{consensus}_{i,t}}{\sigma_{\text{hist},i}}
$$

In the free-data tier, consensus is proxied by the rolling mean: 

$$
\text{consensus}_{i,t} \approx \frac{1}{H}\sum_{\tau=1}^{H}S_{i,t-\tau}
$$

**Step 2 — Composite nowcast index:**
$$S_{i,t} = w^{\text{PMI}}\Delta_i^{\text{PMI}} + w^{\text{CPI}}\Delta_i^{\text{CPI}} + w^{\text{EMP}}\Delta_i^{\text{EMP}}, \quad w = (0.40, 0.30, 0.30)^T$$

Weights reflect empirical predictive content for FX returns: PMI flash releases have highest currency impact (Andersen et al. 2003), followed by inflation and employment.

**Step 3 — EMA as a Kalman-like filter:**

The EMA 

$$\hat{S}_{t} = \alpha S_t + (1-\alpha)\hat{S}_{t-1}$$

with $\alpha = 2/(\tau+1)$ is the steady-state solution of a scalar Kalman filter with signal-to-noise ratio $q = \alpha^2/(1-(1-\alpha)^2)$. It provides an optimal linear estimate of the forward-curve expectation under a local-level model:

$$
S_t = \mu_t + \varepsilon_t, \quad \mu_t = \mu_{t-1} + \eta_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon),\; \eta_t \sim \mathcal{N}(0,\sigma^2_\eta)
$$

**Step 4 — Divergence signal:**
$$\alpha_i^{\text{MGD}} = \frac{S_{i,t} - \hat{S}_{i,t}}{\max(\sigma^{60}_i, \varepsilon)}$$

This is precisely the *innovation* $S_{t|t} - S_{t|t-1}$ of the Kalman filter, normalised by the 60-day rolling standard deviation. Under the EMH null, $\mathbb{E}[\alpha^{\text{MGD}}] = 0$; the alternative hypothesis is that institutional flow lags create predictable response.

### 4. Gaussian Rank Normalisation

Signal scores are mapped to normal quantiles to ensure cross-sectional distributional stability (crucial for portfolio optimisers that assume elliptical return distributions):

$$r_{i,t} = \Phi^{-1}\!\left(\frac{\text{rank}(z_{i,t})}{N+1}\right)$$

This is a non-parametric transformation that: (1) eliminates sensitivity to outliers, (2) ensures the marginal distribution of scores is $\mathcal{N}(0,1)$ by construction, and (3) preserves the ordinal ranking while standardising the scale.

### 5. Sharpe Ratio Waterfall — Full Derivation

Pre-cost Sharpe floor for systematic macro at 4× daily frequency:

$$
\text{SR}_{\text{gross}} \geq 2.0 \implies \text{SR}_{\text{net}} \approx 1.15
$$

| Haircut Source | Mechanism | Magnitude |
|---------------|-----------|-----------|
| Bid-ask spread | $\frac{1}{2}\text{spread} \times \text{turnover} \times 252 \times 4$ | −0.40 SR |
| IS bias + overfitting | Lopez de Prado (2018): $`\text{SR}_{\text{live}} = \text{SR}_{\text{IS}} \times (1 - \hat{\rho})`$ | −0.30 SR |
| Market impact | $\lambda \sqrt{P \cdot \sigma \cdot \text{ADV}^{-1}}$ (Almgren-Chriss) | −0.15 SR |

Statistical significance: $t = \text{SR} \times \sqrt{T \times 4} \geq 3.0$.  
For $T=252$ days, 4× daily: $t = 2.0 \times \sqrt{1008} \approx 63.5$ ✓

**Six-month walk-forward target:** $\text{SR}_{\text{walk-fwd}} > 0.70$ (lower floor accounts for limited OOS history in first 6 months; per Proprietary Trading Firm plan KPI).

### 6. Deflated Sharpe Ratio (DSR)

From Bailey & Lopez de Prado (2014), the DSR corrects for selection bias when $M$ strategy variants are tested:

$$\text{DSR} = \Phi\!\left(\frac{(\hat{\text{SR}} - \text{SR}^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3\hat{\text{SR}} + \tfrac{\hat{\gamma}_4-1}{4}\hat{\text{SR}}^2}}\right)$$

where:

$$
\hat{\gamma}_3, \hat{\gamma}_4
$$

are sample skewness and excess kurtosis of the P&L series, and the expected maximum SR under $M=2$ trial configurations:

$$
\text{SR}^* \approx (1-\gamma_{\text{Euler}})\Phi^{-1}\!\left(1-\tfrac{1}{M}\right) + \Phi^{-1}\!\left(1 - \tfrac{1}{M e^{1/2}}\right)^{1/2}
$$

### 7. Ornstein-Uhlenbeck Half-Life for Signal Retirement

Signal IC series is modelled as an Ornstein-Uhlenbeck process:
$$d\text{IC}_t = \kappa(\mu - \text{IC}_t)\,dt + \sigma\,dW_t$$

Discretised OLS regression on the IC series:

$$
\Delta\text{IC}_t = -\hat{\kappa}\,\text{IC}_{t-1} + \varepsilon_t
$$

Half-life: $T_{1/2} = \ln 2 / \hat{\kappa}$. Pre-agreed retirement gates:

| Condition | Action |
|-----------|--------|
| $T_{1/2} < 21$ days | 🚨 **Retire** — signal decaying faster than minimum viable persistence |
| $T_{1/2} \in [21, 63]$ days | ✅ **Alive** — healthy decay regime |
| $T_{1/2} > 63$ days | ⚠️ **Review** — signal may be stale / overfitted to a single regime |
| $\text{IC} < 0.02$ | 🚨 **Retire** — below IC floor |
| $\text{ICIR} < 0.50$ | ⚠️ **Reduce allocation** |

---
---

## ML and Statistical Methodology

[🔝 Back to Top](#-table-of-contents)

### Statistics-First Philosophy

Following Proprietary Trading Firm's research culture: every technique must earn its place through statistical rigor, not novelty.

**Linear methods (primary):** OLS regression for Granger causality and OU half-life estimation. Ledoit-Wolf shrinkage (Ledoit & Wolf 2004) for covariance estimation — analytically optimal shrinkage intensity $\hat{\alpha}^*$ minimises the Frobenius norm $\|\hat{\Sigma} - \Sigma\|_F^2$ under a random matrix theory framework. No hyperparameter search needed.

**Non-linear methods (secondary, where justified):**
- Gram-Schmidt residualisation is equivalent to partial regression — a linear projection. Non-linear extensions (kernel Gram-Schmidt) are available but require $T \gg N^2$ observations to be statistically stable.
- The MAD-based robust z-score is a linear statistic with high breakdown point — more appropriate than RANSAC or Huber regression at this sample size.
- The MBB (Moving Block Bootstrap) is a non-parametric resampling method that makes no distributional assumptions beyond stationarity and weak dependence — appropriate for macro time-series with long-memory.

**Deep Learning — where it genuinely adds value (not yet, explicitly by design):**
At a 4× daily frequency with $T \approx 1,000$ observations per year, the signal-to-noise ratio is too low for transformer-based sequence models (which typically require $T > 10,000$ per asset to avoid severe overfitting). The FLOAM framework shows that IC = 0.04 with depth Breadth = 1008 already achieves IR ≈ 1.27 — adding deep learning without commensurate data depth risks spending the overfitting budget on model complexity rather than genuine predictive content.

**The regime for ML:** When daily-frequency alternative data (satellite imagery of commodity warehouses, NLP on CB minutes, high-frequency order flow) is integrated (Month 1–2 of the six-month plan), an ensemble of LightGBM + sparse linear (Elastic Net) as a stacking layer over the ISCF/MGD scores is the next planned step. This exploits gradient-boosted trees' ability to capture non-linear interactions between the basis z-score and inventory-level thresholds, while Elastic Net handles the high-dimensional surprise cross-asset matrix.

### Time-Series Considerations

**Non-stationarity:** All signals are computed cross-sectionally at each $t$ — the cross-sectional ranking operation is stationarity-preserving even if individual asset price series are I(1). IC time series are tested for stationarity via ADF (H0: unit root) before half-life estimation.

**Autocorrelation correction:** HAC Newey-West with bandwidth $L = \lfloor 4(T/100)^{2/9}\rfloor$ (Andrews 1991 optimal) prevents downward-biased standard errors in Granger F-tests.

**Multiple testing:** With $M=2$ signals, Bonferroni FWER correction gives $\alpha^* = 0.025$. Benjamini-Hochberg FDR controls the expected false discovery rate. Both are computed and reported.

---
---

## Data Provider Architecture

[🔝 Back to Top](#-table-of-contents)

```python
# TODAY (free tier):
provider = get_iscf_provider(mode="yfinance")
data: ISCFMarketData = provider.fetch(start="2018-01-01", end="2024-12-31")

# DAY 1 AT Proprietary Trading Firm (zero code change):
provider = get_iscf_provider(mode="proprietary", api_key=os.environ["HLS_API_KEY"])
data: ISCFMarketData = provider.fetch(start="2018-01-01", end="2024-12-31")
```

**Design-by-Contract guarantees** enforced by `validate()` on every provider:

| Contract | Postcondition |
|----------|--------------|
| `spot > 0` | No negative prices |
| `rvol > 0` | No zero volatility |
| `macro_beta ∈ [0, 1]` | Valid beta range |
| Shape consistency | All arrays `(T, N)` with identical `T` |
| No NaN | All fields forward-filled before return |

| Provider | Data Source | Commodity Proxy | FX + Macro Proxy |
|----------|------------|----------------|-----------------|
| `YFinanceISCFProvider` | yfinance | USO/BNO (crude), UNG (gas), CPER/GLD/SLV | — |
| `YFinanceMGDProvider` | yfinance + FRED | — | EUR/GBP/JPY/AUD/CAD + ISM/CPI/NFP |
| `HLSISCFProvider` | LME/CME + freight | Direct prompt-date spreads + inventory | — |
| `HLSMGDProvider` | Bloomberg + CB | — | Actual vs. consensus surprises + FX forwards |

---
---

## Causal Validation Framework

[🔝 Back to Top](#-table-of-contents)

```
[ Signal ] ─► Step 1: VARX Granger (HAC) ─► Step 2: CMI (α retained≥50%) ─► Step 3: DoWhy ─► γ
                    │ FAIL → REJECT (γ=0)      │ FAIL → BETA_PROXY (γ=0.30)   │ FAIL → REJECT
                    │                           │                               │ PASS → γ=0.95
```

**Step 1 — VARX Granger with HAC correction:**  
$H_0: B_1 = \cdots = B_p = 0$ in $Y_t = \sum A_k Y_{t-k} + \sum B_k X_{t-k} + CZ_t + \varepsilon_t$.  
HAC F-statistic prevents false rejections from autocorrelated residuals.  
Exogenous $Z_t$: session dummies (Asia/London/NY AM/NY PM) + VIX proxy — the intraday confounders identified in `CAUSAL_STACK_EXPLAINED.md`.

**Step 2 — Conditional Independence (CMI proxy):**  
$\psi = |\rho_{\text{partial}}| / |\rho_{\text{raw}}|$ — alpha retained after partialling out session/VIX confounders.  
If $\psi < 0.50$: signal is $> 50\%$ macro beta → route to BETA_PROXY basket.

**Step 3 — DoWhy Placebo + Policy Invariance:**  
Placebo:

$$
p_{\text{placebo}} = B^{-1}\sum \mathbb{1}(|\hat{\theta}_b| \geq |\hat{\theta}_{\text{orig}}|) > 0.05
$$

Policy: Moving Block Bootstrap (block=20) across regime split → structural stability.

**Causal confidence factor $\gamma$:**

| Outcome | γ | Portfolio Behaviour |
|---------|---|-------------------|
| All 3 steps pass | 0.95 | Full target weight; 4× daily rebalancing |
| Granger+CMI pass, policy fail | 0.30 | 30% target weight; 2× daily max |
| Placebo fails | 0.00 | Suspended; zero allocation |

---
---

## 4× Daily Rebalancing Engine

[🔝 Back to Top](#-table-of-contents)

Proprietary Trading rebalances ~4× daily. The rebalancing engine runs at each session:

| Session | UTC Time | Market Context |
|---------|---------|---------------|
| ASIA | 00:00 | Tokyo/Singapore open; JPY/AUD pairs most liquid |
| LONDON | 07:00 | European rates and FX; most liquid session for G10 |
| NY_OPEN | 12:00 | US data releases; highest ISCF/MGD signal frequency |
| NY_CLOSE | 17:00 | Position-squaring; FX forward fixing |

**Target weight construction:**
$$w_i^{\text{target}} = \gamma \cdot w_i^{\text{HRP}}, \quad w_i^{\text{HRP}} \propto \frac{|r_i|}{\sum_j |r_j|}$$

**Transaction cost model:**
$$\text{TC}(bps) = \frac{\text{spread}}{2} \cdot \text{turnover} \cdot 10^4 + \gamma_{\text{impact}} \cdot \sqrt{\frac{\text{turnover}}{\text{ADV}}}$$

**Go/No-Go kill criteria (pre-agreed, per 5-step falsification protocol):**
- $\gamma < 0.25$: causal validation failed → suspend
- Turnover $> 50\%$: split across 2 sessions  
- $\text{TC} > 50\% \times \text{expected gross alpha}$: hold

---
---

## Signal Health Monitoring and Retirement

[🔝 Back to Top](#-table-of-contents)

The `hls-alpha hls-monitor` command runs at each of the 4 daily session crons via GitHub Actions and produces:

1. **Per-signal JSON** (`artifacts/iscf_health.json`, `artifacts/mgd_health.json`)
2. **KPI Markdown report** (`artifacts/KPI_REPORT.md`) posted to `$GITHUB_STEP_SUMMARY`
3. **Exit code 1** if any retirement criterion is triggered

**Retirement decision tree:**
```
IC < 0.02 ──────────────────────────────────────────────► RETIRE
ICIR < 0.50 ──────────────────────────────────────────► REDUCE
T½ < 17.85d (= 21 × 0.85) ──────────────────────────► RETIRE
T½ ∉ [21, 63] ────────────────────────────────────────► REVIEW
SR < 0.70 (walk-fwd floor) ──────────────────────────► REDUCE
Placebo p < 0.05 ─────────────────────────────────────► SUSPEND
All criteria pass ────────────────────────────────────► LIVE ✓
```

When a signal is retired, the research cycle restarts from Step 1 of the 5-step falsification protocol with a new hypothesis.

---
---

## Six-Month Deployment Plan

[🔝 Back to Top](#-table-of-contents)

| Month | Milestone | KPI Gate |
|-------|-----------|---------|
| 1 | Framework audit + LME/CME/FRED data ingestion; baseline Sharpe decay | Decay baseline computed |
| 2 | ISCF+MGD construction + Gram-Schmidt; orthogonality $`R^2 < 0.15`$ | $`\| \rho \| < 0.39`$ with all baseline factors |
| 3 | CPCV (45 paths) + DSR + DoWhy causal stack; FDR control | DSR > 0; all 3 causal steps pass |
| 4 | Black-Litterman with γ priors + TCA; 4× daily rebalancing live | γ-weighted SR > 1.0 paper-trade |
| 5 | Shadow paper-trading; fill rate audit; agentic research pipeline | Fill rate > 95%; slippage < TCA model |
| 6 | Production launch; health dashboard; KPI sign-off | Walk-fwd SR > 0.70; FDR ↓ ≥ 30% |

---
---

## Notebook Research and Plots

[🔝 Back to Top](#-table-of-contents)

### `research.ipynb` — Synthetic panel research

| Plot | File | Description |
|------|------|-------------|
| ISCF IC + PnL | `plots/iscf_ic_pnl.png` | Rolling 60d MA IC + cumulative PnL + drawdown |
| MGD IC + PnL | `plots/mgd_ic_pnl.png` | Rolling 60d MA IC + cumulative PnL + drawdown |
| Causal stack | `plots/causal_validation.png` | 4-bar chart: Granger/CMI/Placebo/Policy per signal |
| Orthogonality | `plots/orthogonality.png` | R² heatmap: ISCF+MGD vs Trend/Momentum/Carry |
| Sharpe waterfall | `plots/sharpe_waterfall.png` | Gross→TC→Overfit→Slippage→Net SR |

### `backtest.ipynb` — Real market data (yfinance + FRED)

Full pipeline on live data: ISCF (ETF price proxies for commodity futures basis) + MGD (G10 FX + FRED macro surprises). Includes CPCV (45 paths), DSR, BH/Bonferroni multiple testing, causal validation, γ-weighted portfolio construction. Graceful fallback to synthetic data if network unavailable.

---
---

## Build Compile and Run Instructions

[🔝 Back to Top](#-table-of-contents)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13 | [python.org](https://www.python.org/downloads/) |
| uv | ≥ 0.5 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| CMake | ≥ 3.28 | `apt install cmake` |
| g++ | 14 | `apt install g++-14` |
| Ninja | any | `apt install ninja-build` |

### Local — Python 3.13 (uv)

```bash
git clone https://github.com/<org>/hls-alpha-signals.git && cd hls-alpha-signals
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[dev]"

hls-alpha hls-run --n-assets 8 --n-periods 2000          # Synthetic pipeline
hls-alpha hls-live --start 2018-01-01 --end 2024-12-31   # Real data (yfinance)
hls-alpha hls-monitor --output-dir artifacts              # KPIs + rebalancing
hls-alpha benchmark --n-assets 8 --n-reps 5000           # C++ vs Python timing
pytest tests/python/ -v -n auto                           # Full test suite
```

### Local — C++26 (CMake)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
cmake --build build --parallel $(nproc)
cmake --install build --prefix .   # Installs .so next to Python package
cd build && ctest --output-on-failure
```

### Docker

```bash
docker build -t hls-alpha:latest .
docker run --rm hls-alpha:latest hls-alpha --version
docker run --rm -v $(pwd)/artifacts:/app/artifacts hls-alpha:latest \
  hls-alpha hls-monitor --output-dir /app/artifacts
```

---
---

## GitHub Actions CI/CD

[🔝 Back to Top](#-table-of-contents)

### `Running Manually`

Example trigger github actions workflow for the branch **hls_trading_alpha_research_six_month_plan**.

#### Windows
```
cd .github\workflows

gh auth login

gh workflow run ci.yml --ref hls_trading_alpha_research_six_month_plan
```

#### Linux
```
cd .github/workflows

gh auth login

gh workflow run ci.yml --ref hls_trading_alpha_research_six_month_plan
```

**You should see the output like so:**

```text
✓ Created workflow_dispatch event for ci.yml at hls_trading_alpha_research_six_month_plan
```

### `Fetch the artifacts`

Fetch the artifacts generated by the github actions workflow for the branch **hls_trading_alpha_research_six_month_plan**.

#### Windows
```
cd .github\workflows

gh auth login

gh run list --workflow=ci.yml -L 2

gh run download <run-id>
# gh run download 27243308768
```

#### Linux
```
cd .github/workflows

gh auth login

gh run list --workflow=ci.yml -L 2

gh run download <run-id>
# gh run download 27243308768
```

**You should see the output like so:**

```text
STATUS  TITLE                  WORKFLOW            BRANCH              EVENT              ID           ELAPSED  AGE
✓       CI — Citadel Alpha...  CI — Citadel Al...  hls_trading_alp...  workflow_dispatch  27243308768  2m47s    about 50 minute...
✓       CI — Citadel Alpha...  CI — Citadel Al...  hls_trading_alp...  workflow_dispatch  27241806356  3m5s     about 1 hour ago
```

### Description of the actions

**4× daily cron schedule** mirrors Proprietary Trading Firm 's rebalancing frequency:

| UTC Time | Session | GitHub Actions Cron |
|---------|---------|---------------------|
| 07:00 | London Open | `0 7 * * 1-5` |
| 12:00 | NY Open | `0 12 * * 1-5` |
| 17:00 | London Close | `0 17 * * 1-5` |
| 21:00 | NY Close / Asia | `0 21 * * 1-5` |

**Jobs:**

| Job | Depends On | What Runs |
|-----|-----------|-----------|
| `python-build-test` | — | `pytest` (35 tests) + coverage XML |
| `cpp-build-test` | — | CMake/Ninja Release + `ctest` (GTest) |
| `signal-health-monitor` | both | Build C++ .so → `hls-alpha hls-monitor` → `generate_kpi_report.py` → `$GITHUB_STEP_SUMMARY` |
| `benchmark` | both pass | `hls-alpha benchmark` timing artifact |
| `docker-build` | Python passes | Multi-stage build + smoke test |

The `signal-health-monitor` job: (1) builds the C++26 nanobind extension, (2) installs it so the Python CLI uses the compiled hot-path, (3) runs `hls-alpha hls-monitor` which executes the full Python research stack, (4) generates a markdown KPI report posted inline to the GitHub Actions run summary.

---
---

## Dissertation

[🔝 Back to Top](#-table-of-contents)

Full academic-style dissertation with first-principles mathematical derivations, causal framework, regime analysis, and six-month deployment plan:

- **Markdown:** [`dissertation/README.md`](dissertation/README.md)
- **LaTeX:** [`dissertation/hls_alpha_dissertation.tex`](dissertation/hls_alpha_dissertation.tex)
- **PDF:** [`dissertation/hls_alpha_dissertation.pdf`](dissertation/hls_alpha_dissertation.pdf)

```bash
cd dissertation && pdflatex hls_alpha_dissertation.tex && pdflatex hls_alpha_dissertation.tex
```

---
---

## References

[🔝 Back to Top](#-table-of-contents)

- Andersen et al. (2003) — *Micro Effects of Macro Announcements*, AER
- Andrews (1991) — *Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation*, Econometrica
- Bailey & Lopez de Prado (2014) — *The Deflated Sharpe Ratio*, JPM
- Fama & French (1988) — *Business Cycles and the Behavior of Metals Prices*, JF
- Grinold (1989) — *The Fundamental Law of Active Management*, JPM
- Ledoit & Wolf (2004) — *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*, JMVA
- Lopez de Prado (2018) — *Advances in Financial Machine Learning*, Wiley
- Muth (1961) — *Rational Expectations and the Theory of Price Movements*, Econometrica
- Newey & West (1987) — *A Simple Positive Definite HAC Covariance Matrix*, Econometrica
- Pearl (2009) — *Causality: Models, Reasoning and Inference*, CUP
- Working (1949) — *The Theory of Price of Storage*, AER
