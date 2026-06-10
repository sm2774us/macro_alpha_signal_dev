# 📐 Quant Research Field Manual: Industry-Standard "Goldilocks" Constants

> **Audience:** Senior Quant Researchers — Systematic Macro | Citadel · Millennium · Jane Street · Two Sigma · DE Shaw
>
> **Purpose:** Empirical constants that balance model fidelity against noise, transaction costs, liquidity, and statistical limits.  
> **Regime:** These values are calibrated for *liquid macro markets* (G10 FX, Rates, Equity Index, Commodities).

---

## 📋 Synopsis

Every practitioner in systematic macro eventually converges on the same small set of "just right" numbers — not because theory demands them, but because *the market enforces them*. Set your EWMA decay too high and you chase noise; too low and you hold stale risk. Sample intraday returns too fast and microstructure dominates; too slow and you lose signal. These are the empirical constants that survive the friction of real execution: bid-ask spread, funding costs, regime shifts, and risk limits.

This manual covers **five pillars**:

| # | Pillar | Core Constant(s) |
|---|--------|-----------------|
| 1 | [Mean-Reversion & Decay Rates](#1--mean-reversion--decay-rates) | $\kappa \approx 0.01\ \text{day}^{-1}$ |
| 2 | [Volatility & Risk Modeling](#2--volatility--risk-modeling) | $\lambda = 0.94$, $\alpha \in \{0.05, 0.01\}$, $\tau \in [2,3]$ |
| 3 | [Signal Processing & Microstructure](#3--signal-processing--microstructure) | $\gamma = 0.5$, $\Delta t = 5\text{ min}$, $\rho \approx 0.05\text{–}0.15$ |
| 4 | [Backtesting & Portfolio Construction](#4--backtesting--portfolio-construction) | $\text{SR} = 2.0$, $N = 252$, $w_{\max} = 0.05$ |
| 5 | [Extended Constants: The Full Arsenal](#5--extended-constants-the-full-arsenal) | $\text{CR} \geq 0.5$, $\beta_{\text{hedge}} \approx 0.95\text{–}1.05$, $\nu \approx 4\text{–}6$, … |

---
---

[↩️ Back to README.md](./README.md)

---
---

## 📑 Table of Contents

- [1 · Mean-Reversion & Decay Rates](#1--mean-reversion--decay-rates)
  - [1.1 Options & Volatility — Daily Theta](#11-options--volatility--daily-theta-κ--001)
  - [1.2 Macroeconomic Processes — OU Mean Reversion](#12-macroeconomic-processes--ou-mean-reversion-κ--001)
  - [1.3 Why κ ≈ 0.01 is the Sweet Spot](#13-why-κ--001-is-the-sweet-spot)
- [2 · Volatility & Risk Modeling](#2--volatility--risk-modeling)
  - [2.1 EWMA Decay — λ = 0.94](#21-ewma-decay--λ--094)
  - [2.2 VaR Confidence Levels — α = 0.05 / 0.01](#22-var-confidence-levels--α--005--001)
  - [2.3 Z-Score Entry Threshold — τ ∈ [2, 3]](#23-z-score-entry-threshold--τ--2-3)
- [3 · Signal Processing & Microstructure](#3--signal-processing--microstructure)
  - [3.1 The Square-Root Law — γ = 0.5](#31-the-square-root-law--γ--05)
  - [3.2 Intraday Sampling — Δt = 5 min](#32-intraday-sampling--Δt--5-min)
  - [3.3 Cross-Asset Correlation Baseline — ρ ≈ 0.05–0.15](#33-cross-asset-correlation-baseline--ρ--005015)
- [4 · Backtesting & Portfolio Construction](#4--backtesting--portfolio-construction)
  - [4.1 Pre-Cost Sharpe Floor — SR = 2.0](#41-pre-cost-sharpe-floor--sr--20)
  - [4.2 Annualisation Factor — N = 252](#42-annualisation-factor--n--252)
  - [4.3 Single-Position Limit — w_max = 5%](#43-single-position-limit--w_max--5)
- [5 · Extended Constants: The Full Arsenal](#5--extended-constants-the-full-arsenal)
  - [5.1 Fat Tails — Student-t DoF ν ≈ 4–6](#51-fat-tails--student-t-dof-ν--46)
  - [5.2 Delta Hedging Bandwidth — Δ-band ≈ 1–2%](#52-delta-hedging-bandwidth--Δ-band--12)
  - [5.3 Beta Hedge Ratio — β ≈ 0.95–1.05](#53-beta-hedge-ratio--β--0951-05)
  - [5.4 Calmar Ratio Floor — CR ≥ 0.5](#54-calmar-ratio-floor--cr--05)
  - [5.5 ADV Participation Rate Cap — p ≤ 10–20%](#55-adv-participation-rate-cap--p--1020)
  - [5.6 Covariance Matrix Shrinkage — α_LW ≈ 0.1–0.3](#56-covariance-matrix-shrinkage--α_lw--01030)
  - [5.7 Half-Life of Alpha Decay — T_½ ≈ 21–63 days](#57-half-life-of-alpha-decay--t_½--2163-days)
  - [5.8 Volatility Regime Threshold — VIX ≈ 20](#58-volatility-regime-threshold--vix--20)

---

## 1 · Mean-Reversion & Decay Rates

### 1.1 Options & Volatility — Daily Theta ($\kappa \approx 0.01$)

In a macro fund context, $\kappa \approx 0.01\ \text{day}^{-1}$ most commonly models **daily option theta** — the fraction of premium eroding per calendar or trading day.

$$\theta_{\text{daily}} = \kappa \cdot V_0 \approx 0.01 \times V_0$$

where $V_0$ is the current option premium. Annualised:

$$\theta_{\text{annual}} = \kappa \cdot V_0 \cdot N = 0.01 \times V_0 \times 365 \approx 3.65 \cdot V_0$$

| Day | Remaining Premium ($V_0 = \\$100$) |
|-----|-----------------------------------|
| 0 | \$100.00 |
| 30 | \$74.08 |
| 69 | \$50.00 ← **half-life** |
| 100 | \$36.79 |
| 252 | \$8.11 |

> 💡 **Practitioner note:** Macro funds structure option books around **vega-neutral, theta-positive** overlays on sovereign bond and FX positions. A $\kappa = 0.01$ carry burn rate fits neatly within a 1% daily risk budget.

---

### 1.2 Macroeconomic Processes — OU Mean Reversion ($\kappa \approx 0.01$)

When $\kappa$ parameterises the **Ornstein–Uhlenbeck (Vasicek) process** for rates or inflation:

$$dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t$$

$\kappa = 0.01$ means the process closes **1% of the gap to $\mu$ per day**.

**Half-life of a macro shock:**

$$T_{1/2} = \frac{\ln 2}{\kappa} = \frac{0.6931}{0.01} \approx \boxed{69\ \text{days}}$$

```
Shock Decay (κ = 0.01 per day)
Gap to μ
100% ─┐
      │╲
 75%  │  ╲
      │    ╲
 50%  │──────●── Day 69 (half-life)
      │        ╲
 25%  │          ╲
      │            ╲__
  0%  └───────────────────────────▶ days
      0   20   40   60   80   100
```

> 💡 Central bank rate cycles, CPI regimes, and supply-chain normalisation all exhibit half-lives of **60–90 days** empirically — directly justifying $\kappa \approx 0.01$.

---

### 1.3 Why $\kappa \approx 0.01$ is the Sweet Spot

```
          Too Slow          Sweet Spot        Too Fast
          κ < 0.001         κ ≈ 0.01          κ > 0.1
             │                  │                │
   ┌──────────────────┬──────────────────┬──────────────────┐
   │  Risk stays open │ Balances mean-   │ Over-trades;     │
   │  too long; large │ reversion speed  │ cost drag kills  │
   │  drawdowns       │ with liquidity   │ alpha            │
   └──────────────────┴──────────────────┴──────────────────┘
```

| Constraint | Why $\kappa = 0.01$ works |
|------------|--------------------------|
| **VaR / DV01 limits** | 1% daily move maps directly to 1-day 99% VaR budget |
| **Liquidity windows** | Macro books reset over ~5–10 days; OU speed consistent |
| **Policy cycles** | FOMC meets every ~45 days; $T_{1/2} \approx 69$ days bridges meetings |

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)

---
---

## 2 · Volatility & Risk Modeling

### 2.1 EWMA Decay — $\lambda = 0.94$

The **JP Morgan RiskMetrics** universal constant. Daily variance estimate:

$$\hat{\sigma}^2_t = \lambda\,\hat{\sigma}^2_{t-1} + (1-\lambda)\,r^2_{t-1}$$

**Effective memory window:**

$$N_{\text{eff}} = \frac{1}{1-\lambda} = \frac{1}{0.06} \approx \boxed{16.7\ \text{days}}$$

The weight assigned to a return $k$ days ago:

$$w_k = (1-\lambda)\,\lambda^k$$

```
EWMA Weight Profile  (λ = 0.94)
Weight
0.060 ─●
0.056 ─ ●
0.053 ─   ●
       …
0.030 ─         ●
       …
0.010 ─                  ●
0.001 ─                            ●
       ├────┼────┼────┼────┼────┼──▶ days ago
       0    5   10   15   20   50
```

**Why not $\lambda = 0.90$ or $\lambda = 0.97$?**

| $\lambda$ | Memory $N_{\text{eff}}$ | Behaviour |
|-----------|------------------------|-----------|
| 0.90 | 10 days | Overreacts; spikes on single prints |
| **0.94** | **17 days** | **Tracks vol clusters; stable in calm** |
| 0.97 | 33 days | Lags vol expansions; slow to de-risk |

> 💡 For weekly rebalancing desks, $\lambda = 0.97$ is occasionally preferred; for daily risk systems, **0.94 is universal**.

---

### 2.2 VaR Confidence Levels — $\alpha \in \{0.05,\, 0.01\}$

$$\text{VaR}_\alpha = \mu - z_\alpha\,\sigma \quad\text{where}\quad z_{0.05} = 1.645,\quad z_{0.01} = 2.326$$

```
Normal P&L Distribution

         ┌────────────────────────────────┐
         │    99% of outcomes here        │
         │  ┌──────────────────────┐      │
         │  │  95% of outcomes     │      │
         │  │    ┌──────────┐      │      │
    ░░░░░│░░│░░░░│██████████│░░░░░░│░░░░░░│
─────────┴──┴────┴──────────┴──────┴──────▶
    │    │                           P&L
  99%   95%
  VaR   VaR
```

| Level | Use Case | Regulatory Basis |
|-------|----------|-----------------|
| $\alpha = 0.05$ | Internal economic capital, daily P&L limits | Basel II / ISDA |
| $\alpha = 0.01$ | Stress testing, margin requirements | Basel III / FRTB |

**Relationship:** **VaR<sub>1%</sub> ≈ 1.41 × VaR<sub>5%</sub>** under normality (ratio = $2.326/1.645$).

> ⚠️ **Tail-risk caveat:** In macro crises (2008, 2020), realised 1% VaR breaches occur 3–5× more frequently than the model predicts — the case for CVaR / Expected Shortfall overlays.

---

### 2.3 Z-Score Entry Threshold — $\tau \in [2,\, 3]$

For a stat-arb or macro mean-reversion signal:

$$z_t = \frac{X_t - \mu}{\sigma}$$

Enter long/short when $|z_t| \geq \tau$, exit when $|z_t| \leq 0.5$.

```
Signal Z-Score & Trade Zones
z
+3 ─── ─ ─ ─ ─ ─ ─ ─ ─ SHORT ENTRY ─ ─ ─ ─ ─ ─
+2 ─── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   │         ↕ NOISE ZONE (no edge)
+0.5 ─  ─ ─ ─ ─ EXIT  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   0  ─────────────────────────────────────────
-0.5 ─  ─ ─ ─ ─ EXIT  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   │         ↕ NOISE ZONE (no edge)
-2 ─── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
-3 ─── ─ ─ ─ ─ ─ ─ ─ ─ LONG ENTRY ─ ─ ─ ─ ─ ─
```

| Threshold $\tau$ | False-positive rate | Opportunity cost |
|-----------------|---------------------|-----------------|
| $\tau = 1.5$ | ~13% | Drowns in transaction costs |
| $\tau = 2.0$ | ~4.6% | ✅ High vol regimes |
| $\tau = 2.5$ | ~1.2% | ✅ Baseline sweet spot |
| $\tau = 3.0$ | ~0.27% | ✅ Low vol / tight spreads |
| $\tau = 4.0$ | ~0.006% | Too rare; misses alpha |

> 💡 **Regime-adaptive rule:** Use $\tau = 2$ when VIX > 25 (wider spreads need more signal), $\tau = 2.5$ in normal regimes.

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)

---
---

## 3 · Signal Processing & Microstructure

### 3.1 The Square-Root Law — $\gamma = 0.5$

Market impact scales with the **square root** of normalised order size:

$$I = \eta \cdot \sigma \cdot \sqrt{\frac{Q}{V_{\text{ADV}}}}$$

where $\sigma$ = daily vol, $Q$ = order size, $V_{\text{ADV}}$ = average daily volume, $\eta \approx 0.1\text{–}0.5$ (asset-class dependent).

**Derivation intuition:**

$$\text{Cost} \propto \sqrt{\text{Fraction of book consumed}}$$

```
Market Impact vs. Order Size  (Square-Root Law)

Impact
  1.0 │                              ●
      │                         ●
  0.7 │                    ●
      │               ●
  0.5 │          ●
      │     ●
  0.3 │  ●
      │●
  0.0 └─────────────────────────────▶
      0   10%  25%  50%  100%  200%
                  Q / V_ADV
```

**Scaling implications for a systematic macro book:**

| Order Size ($Q/V_{\text{ADV}}$) | Approximate Impact (bps) |
|--------------------------------|--------------------------|
| 1% | ~3 bps |
| 5% | ~7 bps |
| 10% | ~10 bps |
| 25% | ~16 bps |
| 100% | ~32 bps |

> ⚠️ **Non-linearity:** Impact is *superlinear* beyond ~30% ADV — the model breaks down. Never plan executions above 20% ADV without an VWAP/TWAP algorithm.

---

### 3.2 Intraday Sampling — $\Delta t = 5\ \text{min}$

The **Epps effect** and **bid-ask bounce** create a noise floor below $\Delta t \approx 5$ min. Above $\Delta t \approx 30$ min, you sacrifice signal density unnecessarily.

**Variance ratio diagnostic:**

$$VR(\Delta t) = \frac{\text{Var}[r_{\Delta t}]}{\Delta t \cdot \text{Var}[r_{1\text{min}}]}$$

$VR \neq 1$ indicates microstructure contamination.

```
Signal Quality vs. Sampling Frequency

                ┌────────────────────────────────────┐
Quality /       │  Microstructure │  Sweet  │ Low    │
Info Ratio      │  noise zone     │  Spot   │ freq   │
                │                 │         │ decay  │
       High ──  │                 │  ●●●●●  │        │
                │                 │ ●     ● │  ●●    │
       Med  ──  │     ●●●         │         │    ●●  │
                │   ●●   ●●       │         │      ● │
       Low  ── ─│●●               │         │        │
                └────────────────────────────────────┘
                 1s  1m  3m  │5m│  15m  30m  1h  4h
                             ▲
                        GOLDILOCKS
```

| Sampling $\Delta t$ | Dominant Effect | Recommended For |
|--------------------|-----------------|-----------------|
| < 1 min | Bid-ask bounce, latency | HFT only |
| 1–3 min | Microstructure noise | Mid-freq with noise filter |
| **5 min** | **Clean returns** | **Systematic macro, stat-arb** |
| 15–30 min | Reduced obs count | Options vol surface fitting |
| 1 hour+ | Event-driven signals | Fundamental macro |

> 💡 **Realized variance estimator** at 5-min: $\hat{\sigma}^2 = \sum_{i=1}^{78} r_i^2$ (78 bars per 6.5-hour US session).

---

### 3.3 Cross-Asset Correlation Baseline — $\rho \approx 0.05\text{–}0.15$

In **normal** (non-crisis) regimes, truly orthogonal liquid assets (e.g., 10Y UST vs. WTI crude vs. AUDJPY carry) exhibit **residual background correlation**:

$$\rho_{\text{baseline}} \approx 0.05\text{–}0.15$$

**Why non-zero?** Common macro factors (USD liquidity, global risk appetite, commodity supply) create unavoidable co-movement even across structurally unrelated assets.

```
Correlation Matrix Heatmap (Normal Regime, Stylised)

           UST  SPX  WTI  GOLD  EURUSD  AUDJPY
UST    │  1.00 ─0.35  0.05 0.20  0.15  ─0.25 │
SPX    │ ─0.35  1.00  0.25 0.10  0.20   0.40 │
WTI    │  0.05  0.25  1.00 0.12  0.18   0.30 │
GOLD   │  0.20  0.10  0.12 1.00  0.25   0.10 │
EURUSD │  0.15  0.20  0.18 0.25  1.00   0.55 │
AUDJPY │ ─0.25  0.40  0.30 0.10  0.55   1.00 │

Off-diagonal baseline ≈ |0.05–0.15| in calm regimes
```

> ⚠️ **Crisis correlation collapse:** In risk-off episodes (2008, Mar 2020), cross-asset correlations spike toward $\rho \approx 0.70\text{–}0.90$. Portfolio diversification *appears* to disappear precisely when you need it most.

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)

---
---

## 4 · Backtesting & Portfolio Construction

### 4.1 Pre-Cost Sharpe Floor — $\text{SR} = 2.0$

The **minimum viable threshold** for a strategy to survive the journey from backtest to live trading:

$$\text{SR}_{\text{gross}} \geq 2.0 \implies \text{SR}_{\text{net}} \approx 1.0 \text{–} 1.2$$

**Sharpe decay waterfall (typical macro strategy):**

```
Sharpe Ratio Waterfall

  Gross SR    Transaction  Overfitting  Live         Net SR
  (Backtest)  Costs        Haircut      Slippage &
                                        Decay
  ┌────────┐
  │  2.0   │
  └───┬────┘
      │ ─0.40 (bid-ask, commissions, financing)
      ▼
  ┌────────┐
  │  1.60  │
  └───┬────┘
      │ ─0.30 (IS bias, data-snooping)
      ▼
  ┌────────┐
  │  1.30  │
  └───┬────┘
      │ ─0.15 (market impact, signal decay in live)
      ▼
  ┌────────┐
  │  1.15  │  ← Acceptable live performance
  └────────┘
```

**Minimum SR thresholds by strategy type:**

| Strategy Type | Pre-Cost SR Floor | Rationale |
|---------------|-------------------|-----------|
| Intraday stat-arb | 2.5–3.0 | High turnover; costs bite hard |
| **Systematic macro (daily)** | **2.0** | **Benchmark standard** |
| Cross-asset carry | 1.5–2.0 | Low turnover; carry offsets cost |
| Trend-following CTA | 0.7–1.0 | Long holding periods; low costs |

**Statistical significance check:**

$$t\text{-stat} = \text{SR} \times \sqrt{T} \geq 3.0$$

For $T = 252$ days: $t = 2.0 \times \sqrt{252} \approx 31.7$ ✅ (highly significant).

---

### 4.2 Annualisation Factor — $N = 252$

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$
$$\mu_{\text{annual}} = \mu_{\text{daily}} \times 252$$
$$\text{SR}_{\text{annual}} = \frac{\mu_{\text{annual}}}{\sigma_{\text{annual}}} = \frac{\mu_{\text{daily}} \times 252}{\sigma_{\text{daily}} \times \sqrt{252}} = \text{SR}_{\text{daily}} \times \sqrt{252}$$

**Why 252 and not 260 or 365?**

```
Calendar Year Breakdown (US Markets)
─────────────────────────────────────
  365 calendar days
─  10 US federal holidays
─   104 weekends (Sat + Sun)
─    ~1 occasional market closure
  ─────────────────────────────────
  ≈  250–252 trading days
```

> 💡 The exact count varies 250–253 depending on year. **252 is the institutional consensus** — it's the median, avoids leap-year edge cases, and is divisible by 4 (useful for quarterly scaling). For non-US assets, some desks use 260 (full weekdays) or 256 ($2^8$, convenient for bit-shifting in legacy systems).

**Scaling ladder:**

| Horizon | Scaling multiplier |
|---------|-------------------|
| Daily → Weekly | $\sqrt{5}$ |
| Daily → Monthly | $\sqrt{21}$ |
| Daily → Quarterly | $\sqrt{63}$ |
| Daily → Annual | $\sqrt{252}$ |

---

### 4.3 Single-Position Limit — $w_{\max} = 5\%$

The **regulatory and risk ceiling** for any single issuer, instrument, or correlated cluster:

$$w_i \leq w_{\max} = 0.05 \quad \forall i$$

**Concentration risk visualisation:**

```
Portfolio Weight Limits

Max Drawdown from Single Position Blowup

 w = 10% │████████████████████████████│ ─10% (unacceptable)
 w = 7%  │████████████████████        │ ─7%
 w = 5%  │██████████████              │ ─5%  ← LIMIT
 w = 3%  │████████                    │ ─3%  ← Conservative
 w = 1%  │████                        │ ─1%  ← Diversified book
         └────────────────────────────┘
```

**Firm-level context:**

| Constraint Source | Typical $w_{\max}$ | Scope |
|------------------|--------------------|-------|
| SEC 5% rule (registered funds) | 5% | Single issuer |
| UCITS regulation | 5% (10% exception) | Single issuer |
| Internal risk policy (top quant funds) | 3–5% | Correlated cluster |
| Macro CTA (concentrated bets) | 8–10% | Directional macro theme |

> 💡 **Cluster-adjusted rule:** In systematic macro, correlated positions (e.g., long EUR/USD + long EUR/GBP + short USD/CHF) are often capped at **5% as a *cluster***, not per-leg.

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)

---
---

## 5 · Extended Constants: The Full Arsenal

*The constants below are equally "institutional" but less frequently documented in textbooks. Every senior quant researcher at a top systematic macro shop is expected to internalise these.*

---

### 5.1 Fat Tails — Student-t DoF $\nu \approx 4\text{–}6$

Macro asset returns are **not Gaussian**. The Student-t distribution with $\nu \approx 4\text{–}6$ degrees of freedom matches empirical tail probabilities far better:

$$f(x;\nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

**Excess kurtosis:** $\kappa_4 = \frac{6}{\nu - 4}$ for $\nu > 4$

| $\nu$ | Excess Kurtosis | Tail multiplier vs. Normal (4-sigma) |
|-------|----------------|--------------------------------------|
| 3 | ∞ | ∞ |
| 4 | 6.0 | ~15× |
| **5** | **2.0** | **~6×** |
| **6** | **1.5** | **~4×** |
| 30 | ~0.2 | ~1.2× |
| ∞ (Normal) | 0 | 1× |

> 💡 For VaR models, using $\nu = 5$ instead of Gaussian **increases** the 1% VaR by ~30–40% — this is the correct capital buffer in macro trading.

---

### 5.2 Delta Hedging Bandwidth — $\Delta$-band $\approx 1\text{–}2\%$

For delta-hedging vanilla options positions, re-hedging every tick is optimal in theory (Black-Scholes) but destroys P&L in practice. The **Leland (1985) / Wilmott bandwidth rule**:

$$\text{Re-hedge when}\quad |\Delta_t - \Delta_{\text{target}}| \geq \varepsilon \approx 0.01\text{–}0.02$$

**Cost-error tradeoff:**

```
Re-hedge Bandwidth Tradeoff

Hedging error (gamma exposure)
 ▲                     ●
 │                ●
 │           ●
 │      ●
 │  ●──────────────────── Sweet Spot ──
 └──────────────────────────────────▶
   0   0.5%  1%  1.5%  2%  3%  4%
              Re-hedge band (ε)

Transaction costs (bps/day)
 ▲●
 │ ●
 │   ●
 │     ●──────────────────────── Flat ─
 └──────────────────────────────────▶
   0   0.5%  1%  1.5%  2%  3%  4%
```

> 💡 For G10 FX options (tight spreads): $\varepsilon = 1\\%$. For EM or illiquid rates options: $\varepsilon = 2\\%\text{–}3\\%$.

---

### 5.3 Beta Hedge Ratio — $\beta \approx 0.95\text{–}1.05$

For a market-neutral systematic macro strategy, the target beta to the index (SPX, aggregate global equities) should be:

$$\beta_{\text{portfolio}} \in [0.95,\, 1.05]$$

This range accounts for:
- **Estimation error** in rolling beta ($\pm 0.05$ is within 1 standard error for 60-day windows)
- **Hedge slippage** from futures roll costs (~0.02–0.05 per roll)
- **Corporate action noise** in the index constituent weights

$$\hat{\beta} = \frac{\text{Cov}(r_p, r_m)}{\text{Var}(r_m)}, \quad \text{target: } |\hat{\beta} - 1| < 0.05$$

> ⚠️ **Regime drift:** Betas shift materially in stress regimes. Monitor 20-day vs. 60-day beta; divergence > 0.10 signals regime transition.

---

### 5.4 Calmar Ratio Floor — $\text{CR} \geq 0.5$

$$\text{CR} = \frac{\text{CAGR}}{\text{Max Drawdown}}$$

| $\text{CR}$ | Interpretation | Institutional View |
|-------------|---------------|-------------------|
| < 0.3 | Poor risk-adjusted returns | Strategy review required |
| 0.3–0.5 | Marginal | Acceptable only for diversifying overlays |
| **0.5–1.0** | **Good** | **Production standard** |
| 1.0–2.0 | Excellent | Flagship strategy territory |
| > 2.0 | Exceptional | Regime-limited; likely overfitted |

> 💡 Pair with **Sterling Ratio** (CAGR / Avg Drawdown) for a smoother measure less sensitive to single worst drawdown events.

---

### 5.5 ADV Participation Rate Cap — $p \leq 10\text{–}20\\%$

$$p = \frac{Q_{\text{order}}}{V_{\text{ADV}}} \leq \begin{cases} 10\\% & \text{(liquid G10 macro)} \\ 5\\% & \text{(illiquid EM or small-cap)} \end{cases}$$

Exceeding $p = 20\\%$ on a single day triggers:

1. **Price impact** well beyond the $\sqrt{p}$ model (superlinear regime)
2. **Information leakage** — the market identifies the large participant
3. **Regulatory scrutiny** in some jurisdictions (MiFID II large trader reporting)

```
Participation Rate vs. Market Impact

    p = 5%   │●        ~7 bps
    p = 10%  │ ●       ~10 bps
    p = 20%  │    ●    ~14 bps  ← HARD LIMIT
    p = 30%  │      ●  ~20 bps (superlinear)
    p = 50%  │          ●  ~35 bps (market-moving)
             └──────────────────────────────▶
```

---

### 5.6 Covariance Matrix Shrinkage — $\alpha_{\text{LW}} \approx 0.1\text{–}0.3$

The **Ledoit-Wolf (2004)** shrinkage estimator:

$$\hat{\Sigma}_{\text{shrunk}} = (1-\alpha)\,\hat{\Sigma}_{\text{sample}} + \alpha\,F$$

where $F$ is the shrinkage target (e.g., constant-correlation matrix, identity, or factor model).

$$\alpha_{\text{LW}} = \frac{\sum_k \delta_k^2}{\sum_k \|\hat{\Sigma}_k - F\|^2}$$

**Practical guidance:**

| Ratio $T/N$ (obs/assets) | Recommended $\alpha$ | Method |
|--------------------------|---------------------|--------|
| > 10 | 0.05–0.10 | Mild shrinkage |
| 3–10 | **0.10–0.30** | **Ledoit-Wolf** |
| 1–3 | 0.30–0.60 | Aggressive shrinkage or factor model |
| < 1 | — | Full factor model required |

> 💡 For a 50-asset macro portfolio with 252 days of history ($T/N \approx 5$), $\alpha \approx 0.15\text{–}0.20$ is the empirical optimum.

---

### 5.7 Half-Life of Alpha Decay — $T_{1/2} \approx 21\text{–}63$ days

All alpha signals decay. The half-life of IC (Information Coefficient) for systematic macro signals:

$$\text{IC}(t) = \text{IC}_0 \cdot e^{-t/\tau}, \quad T_{1/2} = \tau \ln 2$$

| Signal Type | Typical $T_{1/2}$ | Trading Implication |
|-------------|-------------------|---------------------|
| Short-term momentum (rates) | 5–10 days | Daily rebalancing required |
| Cross-asset carry | 21–42 days | Weekly rebalancing sufficient |
| **Macro fundamental** | **42–63 days** | **Monthly rebalancing viable** |
| CTA trend signals | 60–120 days | Slow turnover, low costs |

```
IC Decay Curves

IC(t)/IC(0)
  1.0 ─●
       │╲  ╲  ╲
  0.5  │  ╲  ╲   ╲─ ─ ─ ─ 63d
       │   ╲  ╲─ ─ ─ ─ 42d
       │    ╲─ ─ ─ ─ 21d
  0.1  │          ●     ●     ●
  0.0  └──────────────────────────▶ days
       0    20   40   60   80  100
```

> 💡 A signal with $T_{1/2} < 5$ days in a daily-rebalancing macro book requires Sharpe > 3.0 pre-cost to survive execution drag.

---

### 5.8 Volatility Regime Threshold — VIX $\approx 20$

The **VIX = 20** level separates:

- **VIX < 20:** Calm / risk-on regime → harvest carry, mean-reversion, and spread strategies
- **VIX > 20:** Elevated regime → switch to trend-following, reduce sizing, widen z-score thresholds
- **VIX > 30:** Crisis regime → tail hedges active, max drawdown controls triggered

$$\text{Regime} = \begin{cases} \text{Risk-on} & \text{VIX} < 20 \\ \text{Transition} & 20 \leq \text{VIX} < 30 \\ \text{Crisis} & \text{VIX} \geq 30 \end{cases}$$

```
VIX Regime Map (2000–2024, stylised)

VIX
 80 │                  ●  (2008 GFC peak: 89.5)
    │
 40 │            ●           ●  (COVID: 85.5)
    │          ● ● ●       ● ●
 30 ├─ ─ ─ ─ CRISIS ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    │       ●             ●
 20 ├─ ─ ─ ─ TRANSITION ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    │  ●  ●   ●  ●  ● ●  ●  ●  ●  ●  ●
 12 │
    └──────────────────────────────────────▶ time
```

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)

---
---

## 🗂️ Master Reference Card

| # | Constant | Value | Formula / Rule | Domain |
|---|----------|-------|----------------|--------|
| 1 | Mean-reversion speed | $\kappa \approx 0.01\ \text{day}^{-1}$ | $T_{1/2} = \ln2/\kappa \approx 69\ \text{days}$ | OU rates/FX, option theta |
| 2 | EWMA decay | $\lambda = 0.94$ | $N_{\text{eff}} = 1/(1-\lambda) \approx 17\ \text{days}$ | Daily vol forecasting |
| 3 | VaR confidence | $\alpha \in \{0.05, 0.01\}$ | $z_{5\\%} = 1.645$; $z_{1\\%} = 2.326$ | Risk limits, margin |
| 4 | Z-score entry | $\tau \in [2.0, 3.0]$ | Enter: $\|z\|\geq\tau$ ; Exit: $\|z\|\leq 0.5$ | Stat-arb signal |
| 5 | Market impact | $\gamma = 0.5$ | $I = \eta\sigma\sqrt{Q/V_{\text{ADV}}}$ | Execution cost |
| 6 | Intraday sampling | $\Delta t = 5\ \text{min}$ | 78 bars per US session | HF return estimation |
| 7 | Background correlation | $\rho \approx 0.05\text{–}0.15$ | Cross-asset, normal regime | Portfolio construction |
| 8 | Gross Sharpe floor | $\text{SR} \geq 2.0$ | $t\text{-stat} = \text{SR}\sqrt{T} \geq 3$ | Strategy filtering |
| 9 | Annualisation factor | $N = 252$ | $\sigma_y = \sigma_d\sqrt{252}$ | Return/vol scaling |
| 10 | Position limit | $w_{\max} = 5\\%$ | Per-issuer or correlated cluster | Risk management |
| 11 | Fat-tail DoF | $\nu \approx 4\text{–}6$ | Excess kurtosis $= 6/(\nu-4)$ | VaR, CVaR |
| 12 | Delta hedge band | $\varepsilon \approx p \leq 1\text{–}2\\%$ | Leland bandwidth rule | Options books |
| 13 | Beta hedge tolerance | $\beta \in [0.95, 1.05]$ | $\hat\beta = \text{Cov}(r_p,r_m)/\text{Var}(r_m)$ | Market-neutral |
| 14 | Calmar ratio floor | $\text{CR} \geq 0.5$ | CAGR / Max Drawdown | Drawdown mgmt |
| 15 | ADV participation cap | $p \leq 10\text{–}20\\%$ | $p = Q/V_{\text{ADV}}$ | Execution limits |
| 16 | Shrinkage intensity | $\alpha_{\text{LW}} \approx 0.10\text{–}0.30$ | Ledoit-Wolf (2004) | Cov matrix est. |
| 17 | Alpha decay half-life | $T_{1/2} \approx 21\text{–}63\ \text{days}$ | $\text{IC}(t)=\text{IC}_0 e^{-t/\tau}$ | Signal design |
| 18 | VIX regime threshold | $\text{VIX} \approx 20$ | < 20 risk-on; ≥ 30 crisis | Regime switching |

---

## 📚 Key References

- **JP Morgan RiskMetrics™ Technical Document** (1996) — $\lambda = 0.94$ EWMA standard
- **Ledoit & Wolf (2004)** — "A well-conditioned estimator for large-dimensional covariance matrices" — shrinkage
- **Almgren & Chriss (2001)** — "Optimal execution of portfolio transactions" — square-root impact law
- **Andersen, Bollerslev et al.** — "Answering the Skeptics" (1998) — 5-minute realised variance
- **Leland (1985)** — "Option Pricing and Replication with Transaction Costs" — delta hedge bandwidth
- **Basel III / FRTB** — Regulatory VaR confidence levels
- **ISDA Margin Framework** — Position and concentration limits

---

*Document maintained for internal quant research use. Regime-check constants at every strategy review cycle.*

[🔝 Back to Top](#-quant-research-field-manual-industry-standard-goldilocks-constants)