# 🎯 RESEARCH CONDENSED — Interview Playbook
### ISCF & MGD: Two Orthogonal Systematic Macro Alpha Signals
#### Shaikat Majumdar · HLS Trading Quantitative Researcher Interview

> **Purpose:** Dissertation-defence framing — sound fundamentals, practitioner instinct, honest
> accounting of what works, what breaks, and how to harden it in production.

---

## 📑 Table of Contents

- [§0 · Why Orthogonal Alpha: The Core Thesis](#0--why-orthogonal-alpha-the-core-thesis)
- [§1 · Signal 1 — ISCF: Idiosyncratic Supply Chain Flow](#1--signal-1--iscf-idiosyncratic-supply-chain-flow)
- [§2 · Signal 2 — MGD: Real-Time Macro Growth Divergence](#2--signal-2--mgd-real-time-macro-growth-divergence)
- [§3 · Causal Validation: Why Prediction ≠ Causation](#3--causal-validation-why-prediction--causation)
- [§4 · Statistical Falsification Protocol](#4--statistical-falsification-protocol)
- [§5 · Portfolio Construction & Sizing](#5--portfolio-construction--sizing)
- [§6 · Six-Month Deployment Plan](#6--six-month-deployment-plan)
- [§7 · Caveats, Pitfalls & Real-World Mitigations](#7--caveats-pitfalls--real-world-mitigations)
- [§8 · Data Requirements & Alternative Data Sourcing](#8--data-requirements--alternative-data-sourcing)
- [§9 · Quick-Reference Equation Sheet](#9--quick-reference-equation-sheet)

---

## §0 · Why Orthogonal Alpha: The Core Thesis

[🔝 Back to Top](#-table-of-contents)

### 🗣️ Feynman Version (15 seconds)
> "Information Ratio grows with the **square root** of the number of **independent** bets.
> Adding a second signal that is 95% correlated with your trend signal contributes almost
> nothing — you are not getting new information, just hearing the same thing louder.
> Adding a signal that exploits a completely different physical mechanism **multiplies your
> effective breadth**. ISCF and MGD are designed to be those orthogonal signals."

### The Mathematics (Grinold 1989)

$$\text{IR} = \text{IC} \cdot \sqrt{\text{Breadth}}$$

- **IC** = Information Coefficient = average rank-correlation between signal and realised returns.
  *Non-mathematically: "How often is the signal right, on average?"*
- **Breadth** = number of **independent** bets per year.
  *Non-mathematically: "How many uncorrelated chances do you get to be right?"*

Adding $K$ mutually orthogonal signals:

$$\text{IR}_{\text{total}} = \text{IC} \cdot \sqrt{K \cdot \text{Breadth}_{\text{single}}} = \sqrt{K} \cdot \text{IR}_{\text{single}}$$

*Non-mathematically: each orthogonal signal multiplies your risk-adjusted edge by the square root of K.
Two orthogonal signals give you 41% more IR with zero additional IC requirement — for free.*

```
BREADTH ARITHMETIC — HLS 4× DAILY FRAMEWORK

  IC  = 0.04  (conservative for systematic macro)
  N   = 8 assets (metals/energy + G10 FX)
  Freq = 4× daily × 252 days = 1008 sessions/year

  Single signal:    IR = 0.04 × sqrt(1008) ≈ 1.27
  Two orthogonal:   IR = 0.04 × sqrt(2016) ≈ 1.80   (+42%)
  If ρ = 0.80 (correlated dupe): IR ≈ 1.33           (+5%)

  The difference between orthogonal and correlated is NOT cosmetic.
  It is the difference between a Sharpe of 1.80 and 1.33 — at scale.
```

### Signal Space Audit — What Existing Factors Already Own

```
Factor        Definition                        What it captures
────────────  ────────────────────────────────  ────────────────────────────────
Trend         sign(r_{t-252 : t-21})            Long-run price momentum (12-1m)
Momentum      r_{t-21 : t}                      Short-term continuation/reversal
Carry         r_domestic - r_foreign            Yield advantage of holding asset

UNSPANNED DIMENSIONS (gaps in existing factor space):
  → Physical delivery constraints in futures    ← ISCF targets this
  → Real-time macro data arrival & lag          ← MGD targets this
```

Both signals are **Gram-Schmidt residualised** against the three baseline factors,
enforcing $\langle \hat{\alpha}^{\text{ISCF}}, \mathbf{f}^k \rangle = 0$ for all $k$,
with VIF < 5 and pairwise $R^2 < 0.15$ as hard gates.

---

## §1 · Signal 1 — ISCF: Idiosyncratic Supply Chain Flow

[🔝 Back to Top](#-table-of-contents)

**Universe:** Metals/Energy futures (Copper, WTI, Nat Gas, Gold, Silver)
**Frequency:** 4× daily rebalancing
**Core mechanism:** Physical delivery-premium backwardation orthogonal to roll yield

### 🗣️ Feynman Version
> "When a copper smelter is running low on physical copper and cannot wait for
> futures delivery, they pay a premium to get it *today*. This creates a peculiar
> situation in the futures market: the near-month contract trades **above** the
> far-month contract — what traders call backwardation. This premium is
> measurable, has a predictable mean-reversion pattern, and is invisible to
> pure price-momentum or yield-carry signals. That is the ISCF edge."

### Step 1 — The Economics: Theory of Storage (Working 1949)

No-arbitrage futures pricing for a storable commodity:

$$F(t,T) = S(t) \cdot e^{(r + u - c)(T-t)}$$

- $S(t)$ = spot price today
- $r$ = risk-free rate, $u$ = storage cost (warehousing + insurance)
- $c$ = **convenience yield** — the value of holding the *physical* commodity

*Non-mathematically: the futures price equals today's spot price compounded at the net
cost of carrying the physical commodity forward. When the cost of carry falls below
the convenience of having it now, futures trade below spot — backwardation.*

The convenience yield is **non-linear** in inventory $I_t$:

$$c = g(I_t) = \begin{cases} c_{\min} & I_t > I^* \\ g_0 + g_1/I_t & I_t \leq I^* \end{cases}$$

```
THE HOCKEY-STICK CONVENIENCE YIELD

  c (convenience
  yield)
     │
  high│─────────────\
     │               \
     │                \_______________
  low│
     └──────────────────────────────────
       0         I*                I_t
              (critical         (abundant)
              threshold)

  Below I*: convenience yield explodes → curve into steep backwardation
  Above I*: convenience yield at floor  → contango or flat
  This kink IS the tradeable event: predictable, mean-reverting, physical.
```

### Step 2 — Volatility-Normalised Basis

$$b_{i,t} = \frac{S_{i,t} - F^{\text{def}}_{i,t}}{\max(\sigma^{rv}_{i,t},\ \varepsilon)}$$

where the 20-day realised volatility:

$$\sigma^{rv}_{i,t} = \sqrt{\frac{252}{20} \sum_{h=1}^{20} \left(\log \frac{S_{i,t-h+1}}{S_{i,t-h}}\right)^2}$$

*Non-mathematically: a $2/bbl basis in WTI crude (45% vol) is trivial noise.
The same $2/bbl basis in a low-vol metal is a 2-sigma event. Dividing by
realised vol puts every commodity on a common risk-adjusted footing.*

```
BASIS NORMALISATION — WHY IT MATTERS

  Asset      Raw Basis   RVol    Normalised Basis
  ─────────  ─────────   ─────   ────────────────
  Copper     +$400       18%     +2.22σ  ← SIGNAL
  Nat Gas    +$0.20      45%     +0.44σ  ← noise
  Gold       +$10        12%     +0.08σ  ← flat
```

### Step 3 — Robust Cross-Sectional Z-Score (MAD)

$$z_{i,t} = \text{clip}\!\left(\frac{b_{i,t} - \text{median}_j(b_{j,t})}{\text{MAD}_j(b_{j,t}) + \varepsilon},\ -4,\ +4\right)$$

$$\text{MAD} = \text{median}_j\left|b_j - \text{median}(b)\right|$$

*Non-mathematically: the classic mean-and-standard-deviation normalisation is
destroyed by a single outlier like the LME Nickel +250% squeeze in 2022.
The Median Absolute Deviation has a 50% breakdown point — you can contaminate
half the cross-section with extreme values and the estimator still converges.*

### Step 4 — Idiosyncratic Extraction & Macro-Beta Stripping

$$\alpha^{\text{ISCF,raw}}_{i,t} = \underbrace{\text{sign}(z_{i,t}) \cdot |z_{i,t}|^{1/2}}_{\text{concave response}} \cdot \underbrace{(1 - \beta^{\text{macro}}_{i,t})}_{\text{idiosyncratic mask}}$$

Two design choices:
1. **Square-root dampening** $|\cdot|^{1/2}$: reduces signal kurtosis from ~6 to ~2; implements
   diminishing returns — extreme backwardation is hard to trade at scale (*limits to arbitrage*).
2. **$(1 - \beta^{\text{macro}})$ mask**: strips out the fraction of the signal that is just
   broad commodity risk-off. *Non-mathematically: if copper is backwardated because everything
   risk-off dropped, that is macro beta — not ISCF.*

```
MACRO-BETA STRIPPING — WORKED EXAMPLE

  z_copper = +2.1σ (strong backwardation)
  β_macro  = 0.75  (moves 75% with broad risk index)

  α_ISCF = sign(2.1) × sqrt(2.1) × (1 - 0.75)
          = +1.449 × 0.25  = +0.36   ← modest, genuinely idiosyncratic

  Same signal, β_macro = 0.05:
  α_ISCF = +1.449 × 0.95  = +1.38   ← strong, supply-chain specific
```

### Step 5 — Gram-Schmidt Orthogonalisation

$$\hat{\alpha}^{\text{ISCF}} = \alpha^{\text{ISCF,raw}} - \sum_{k \in \{\text{trend, mom, carry}\}} \frac{\langle \alpha^{\text{ISCF,raw}}, \mathbf{f}^k \rangle}{\langle \mathbf{f}^k, \mathbf{f}^k \rangle} \mathbf{f}^k$$

*Non-mathematically: project out any component that is linearly explainable by the
existing factors. Whatever remains is — by construction — uncorrelated with trend,
momentum, and carry. Think of it as the residual from a regression.*

### Step 6 — Gaussian Rank Normalisation

$$r_{i,t} = \Phi^{-1}\!\left(\frac{\text{rank}(\hat{\alpha}_{i,t})}{N+1}\right)$$

**Non-mathematically: map ordinal ranks to normal quantiles. This throws away
information about magnitude but guarantees the marginal distribution of scores is
$\mathcal{N}(0,1)$ — which is what downstream portfolio optimisers assume.**

---

## §2 · Signal 2 — MGD: Real-Time Macro Growth Divergence

[🔝 Back to Top](#-table-of-contents)

**Universe:** G10 FX forward panel (EUR, GBP, JPY, AUD, CAD vs USD)
**Frequency:** 4× daily rebalancing
**Core mechanism:** Nowcast composite surprise vs. forward-priced growth expectation

### 🗣️ Feynman Version
> "An FX forward price is the market's best guess about where the rate will
> be at expiry, given current interest rate differentials. When a PMI print
> comes in far above consensus, the market updates — but not instantly.
> Pension funds need compliance sign-off to rebalance; options desks need
> to re-hedge delta. This adjustment lag of 1–6 hours is the MGD edge:
> the market knows the news, but the plumbing takes time to clear."

### Step 1 — Rational Expectations Foundation (Muth 1961)

Under rational expectations, the forward curve is an unbiased predictor:

$$\mathbb{E}_t[S_{t+h}] = F_{t,h}$$

The **surprise** is the residual between realised data and consensus:

$$\Delta^k_{i,t} = \frac{\text{actual}^k_{i,t} - \text{consensus}^k_{i,t}}{\sigma^k_{\text{hist},i}}$$

*Non-mathematically: divide the surprise by its historical standard deviation so
that a 0.5% CPI beat in a high-volatility regime is comparable to a 0.1% beat in
a low-volatility regime.*

### Step 2 — Composite Nowcast Surprise Index

$$\mathcal{S}_{i,t} = 0.40 \cdot \Delta\text{PMI}_{i,t} + 0.30 \cdot \Delta\text{CPI}_{i,t} + 0.30 \cdot \Delta\text{EMP}_{i,t}$$

```
WEIGHT RATIONALE (Andersen et al. 2003, post-2010 calibration)

  Release      FX Impact (5-min)  Persistence (6h)  Weight
  ───────────  ─────────────────  ────────────────  ──────
  Flash PMI    Large              High (growth)      0.40
  CPI          Medium             Medium (CB rxn)    0.30
  NFP/EMP      Medium             Medium (Fed)       0.30

  Note: pre-2008, CPI weight was higher (inflation less anchored).
  The 40/30/30 reflects empirical calibration for 2010–2024.
```

### Step 3 — EMA as Steady-State Kalman Filter

Define the forward-curve expectation $\hat{\mu}_t$ via a local-level state-space model:

$$\mathcal{S}_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon)$$
$$\mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2_\eta)$$

The steady-state Kalman gain is:

$$k^* = \alpha = \frac{\sqrt{q^2 + 4q} - q}{2}, \quad q = \frac{\sigma^2_\eta}{\sigma^2_\varepsilon}$$

This is identical to an EMA with $\alpha = 2/(\tau+1)$, $\tau=5$ days:

$$\hat{\mu}_t = \alpha \mathcal{S}_t + (1-\alpha)\hat{\mu}_{t-1}$$

*Non-mathematically: the EMA is not a heuristic smoothing trick — it is provably
the minimum-mean-squared-error linear filter for a signal that evolves slowly
relative to its noise. We use $\tau = 5$ days, implying a signal-to-noise ratio
$q \approx 0.29$: market-priced growth expectations update about 30 basis points
per day relative to their total noise level.*

### Step 4 — Divergence Signal

$$D_{i,t} = \frac{\mathcal{S}_{i,t} - \hat{\mu}_{i,t}}{\max(\sigma^{60}_{i,t}, \varepsilon)}$$

The numerator $\mathcal{S}_{i,t} - \hat{\mu}_{i,t}$ is the **Kalman filter innovation** —
the genuinely unexpected component of today's macro release.

*Non-mathematically: subtract what the market already priced in (the EMA), divide
by the 60-day rolling standard deviation, and what remains is the pure surprise
that markets have not yet fully absorbed.*

```
MGD EXAMPLE — EUR/USD

  EMA expectation:  μ = +0.15 (mild positive priced in)
  Actual composite: S = +0.65 (strong positive surprise)
  60d rolling std:  σ_60 = 0.20

  Innovation = 0.65 - 0.15 = +0.50
  D = 0.50 / 0.20 = +2.5σ

  → BUY EUR/USD; hold until next session (~6 hours)
  → Edge comes from incomplete institutional flow adjustment
```

---

## §3 · Causal Validation: Why Prediction ≠ Causation

[🔝 Back to Top](#-table-of-contents)

### 🗣️ Feynman Version
> "Three things masquerade as alpha: coincidental correlation with a hidden factor,
> a structural regime change that broke the relationship, and pure data mining.
> The causal stack is a three-step filter designed to kill all three — before
> they get anywhere near capital."

```
THREE FAILURE MODES OF SIGNALS

  Mode 1: SPURIOUS CORRELATION
    Signal and returns both driven by hidden Z_t (e.g., VIX risk-off)
    Test: Granger-VARX controlling for Z_t as exogenous
    Death symptom: signal collapses when VIX regime shifts

  Mode 2: STRUCTURAL BREAK
    Exchange rule change, algo regime change, liquidity shift
    Test: Policy invariance via Moving Block Bootstrap
    Death symptom: works pre-2020, fails post-2020

  Mode 3: DATA SNOOPING
    In-sample optimisation masquerading as genuine alpha
    Test: Placebo (replace signal with random noise, compare)
    Death symptom: IS Sharpe = 2.5, OOS Sharpe = -0.1
```

### Step 1 — VARX Granger Causality with HAC Correction

$$Y_t = \sum_{k=1}^{p} A_k Y_{t-k} + \sum_{k=1}^{p} B_k X_{t-k} + C Z_t + \varepsilon_t$$

$H_0: B_1 = \cdots = B_p = 0$ (signal adds nothing above exogenous controls).

HAC-adjusted F-statistic:

$$F = \frac{(RSS_R - RSS_U)/df_1}{\hat{\sigma}^2_{\text{HAC}}}$$

Newey-West variance with Andrews (1991) optimal bandwidth:

$$\hat{\Sigma}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_{j=1}^{L} \left(1 - \frac{j}{L+1}\right)\left(\hat{\Gamma}_j + \hat{\Gamma}_j^T\right), \quad L = \left\lfloor 4\left(\frac{T}{100}\right)^{2/9}\right\rfloor$$

*Non-mathematically: OLS standard errors assume the residuals are independent draws.
Macro time series have autocorrelation — yesterday's shock echoes into today's. Without
HAC correction, standard errors are too small by 2–3×, turning genuine noise into
statistically significant results. With HAC and $L=12$ lags, only real lead-lag
relationships survive.*

Exogenous controls $Z_t$: session dummies (Asia/London/NY AM/NY PM) + VIX proxy.

### Step 2 — Conditional Independence Test (CMI Proxy)

$$\psi = \frac{|\rho_{\text{partial}}|}{|\rho_{\text{raw}}|}$$

where $\rho_{\text{partial}} = \text{corr}(\tilde{X}, \tilde{Y})$ and $\tilde{X}, \tilde{Y}$ are OLS residuals on confounders.

*Non-mathematically: after conditioning on all known risk factors, how much of the
raw predictive content survives? If less than 50% remains, the signal is more than
half beta-proxy — it gets reduced capital, not independent allocation.*

```
CMI DECISION

  ψ ≥ 0.50  → Genuinely idiosyncratic → proceed to Step 3
  ψ < 0.50  → Mostly beta-proxy → γ = 0.30 (reduced capital)
```

### Step 3 — DoWhy Refutation Tests

**Placebo test:** Replace signal with random noise; test whether original coefficient
is statistically distinct from noise distribution:

$$p_{\text{placebo}} = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(|\hat{\theta}_b| \geq |\hat{\theta}_{\text{orig}}|\right) > 0.05$$

**Policy invariance test:** Moving Block Bootstrap across regime split (train/test halves).
MBB resamples blocks of size $b=20$ to preserve autocorrelation structure.

*Non-mathematically for placebo: "I replaced your signal with random numbers. If random
numbers perform as well as your signal, your signal is random." The bar: be statistically
distinguishable from noise at the 5% level.*

### Causal Confidence Factor γ

$$\gamma = \begin{cases} 0.95 & \text{all 3 steps pass} \\ 0.30 & \text{Granger + CMI pass, policy fail} \\ 0.00 & \text{Granger fails OR placebo fails} \end{cases}$$

Capital allocation: $w_i^{\text{target}} = \gamma \cdot w_i^{\text{HRP}}$

```
γ DECISION TREE

  Step 1: Granger VARX ─── p ≥ 0.05 → γ = 0.00 (REJECT)
                       └── p < 0.05 → Step 2

  Step 2: CMI ─── ψ < 0.50 → γ = 0.30 (BETA_PROXY)
               └── ψ ≥ 0.50 → Step 3

  Step 3: DoWhy ─── placebo fails → γ = 0.00 (REJECT)
                ─── policy fails  → γ = 0.30 (BETA_PROXY)
                └── both pass     → γ = 0.95 (LIVE ✓)
```

---

## §4 · Statistical Falsification Protocol

[🔝 Back to Top](#-table-of-contents)

### 4.1 — Combinatorial Purged Cross-Validation (CPCV)

Lopez de Prado (2018): standard $k$-fold CV is **invalid** for financial time series because
adjacent folds share information through autocorrelated returns.

CPCV with $N=10$, $k=2$:

$$\binom{10}{2} = 45 \text{ independent test paths}$$

*Non-mathematically: instead of one train/test split, generate all possible combinations.
The "purging" removes training observations that are temporally adjacent to test observations,
preventing information leakage through the label overlap that plagues financial ML.*

### 4.2 — Multiple Testing Correction

**Bonferroni** (FWER): $\alpha^* = 0.05/M$ — for $M=2$ signals: $\alpha^* = 0.025$.

**Benjamini-Hochberg** (FDR): reject $H_{(i)}$ for $i \leq \max\{j: p_{(j)} \leq j\alpha/M\}$.

*Non-mathematically: if you test 20 signals at 5% significance, you expect 1 false positive
purely by chance. Bonferroni treats every test as if the cost of a false positive is
catastrophic. BH allows for more discoveries but controls the *fraction* of false ones.*

Target: portfolio FDR reduction $\geq 30\%$ by Month 6.

### 4.3 — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)

$$\text{DSR}(\widehat{SR}) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4-1}{4}\widehat{SR}^2}}\right)$$

where $SR^*$ = expected maximum SR from $M$ random trials, $\hat{\gamma}_3$ = P&L skewness,
$\hat{\gamma}_4$ = excess kurtosis.

*Non-mathematically: a backtest SR of 2.0 sounds excellent, but if you tested 50 parameter
combinations to find it, the benchmark is not zero — it is whatever a lucky random search
would deliver. The DSR adjusts for both the selection bias and the fact that non-normal returns
(negative skew, fat tails) make the SR look better than it truly is. DSR > 0.95 means there
is a 95% chance the strategy is genuinely above zero SR.*

```
SHARPE DECAY WATERFALL

  Gross backtest SR (full IS, pre-cost):    2.50
  ─────────────────────────────────────────────
  Transaction costs haircut:               -0.40
  IS-OOS overfitting gap:                  -0.30
  Live slippage + signal decay:            -0.15
  ─────────────────────────────────────────────
  Expected live net SR:                     1.65

  Month-6 KPI gate: walk-forward SR > 0.70
  (conservative floor; CPCV OOS distribution)
```

### 4.4 — Signal Half-Life via OU Regression

Model the IC time series as an Ornstein-Uhlenbeck process:

$$dIC_t = \kappa(\mu - IC_t)\,dt + \sigma\,dW_t$$

OLS discretisation: $\Delta IC_t = -\hat{\kappa}\, IC_{t-1} + \varepsilon_t$

$$T_{1/2} = \frac{\ln 2}{\hat{\kappa}}$$

*Non-mathematically: no alpha lasts forever. The OU process models how quickly IC
decays back to its mean. Think of it as the "half-life of the edge" — how many days
until the signal loses half its predictive power. Too short means the edge is evaporating
faster than we can trade it. Too long means the signal is so slow it is likely stale.*

```
RETIREMENT DECISION TABLE

  Condition              Threshold        Action
  ─────────────────────  ───────────────  ──────────────────
  Half-life too short    T½ < 17.85d      RETIRE IMMEDIATELY
  Half-life healthy      T½ ∈ [21, 63]d  Continue
  Half-life too long     T½ > 63d         Review (stale?)
  IC below floor         IC < 0.02        RETIRE
  ICIR below floor       ICIR < 0.50      Reduce allocation
  Walk-forward SR        SR < 0.70        Reduce allocation
  Placebo fails          p < 0.05         Suspend
```

---

## §5 · Portfolio Construction & Sizing

[🔝 Back to Top](#-table-of-contents)

### 5.1 — Ledoit-Wolf Covariance Shrinkage

$$\hat{\Sigma}^{\text{LW}} = (1 - \hat{\alpha}^*)\hat{\Sigma}^{\text{sample}} + \hat{\alpha}^* \cdot \frac{\text{tr}(\hat{\Sigma})}{N} I$$

*Non-mathematically: the sample covariance matrix from $T=252$ days and $N=50$ assets
has $N(N+1)/2 = 1275$ free parameters estimated from $252 \times 50 = 12{,}600$ data
points — badly underdetermined. The extreme eigenvalues blow up ($\lambda_{\max} \approx 8$,
$\lambda_{\min} \approx 0.02$), making the inverse numerically unstable. Ledoit-Wolf shrinks
extremes toward the mean eigenvalue, analytically. No cross-validation, no hyperparameters.*

### 5.2 — Hierarchical Risk Parity

HRP bypasses the mean-variance requirement for explicit expected returns:

1. Compute Ledoit-Wolf $\hat{\Sigma}^{\text{LW}}$
2. Correlation distance: $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$
3. Ward hierarchical clustering
4. Recursive bisection: allocate proportional to inverse-variance within each branch
5. Cap: $w_{\max} = 5\%$ per position; ADV participation $\leq 10\%$

*Non-mathematically: group assets by how correlated they are. Then allocate risk evenly
between groups, and within each group. The result is diversified without needing a
mean return estimate — which is typically noisier than the covariance estimate.*

### 5.3 — Bayesian Signal Integration (Black-Litterman)

$$\mu^{\text{BL}} = \left[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau\Sigma)^{-1}\Pi + P^T \Omega^{-1} q\right]$$

where $\Omega^{-1} = \text{diag}(\text{DSR}_{\text{ISCF}}, \text{DSR}_{\text{MGD}}) \cdot \gamma$.

*Non-mathematically: the DSR and $\gamma$ factor become the **confidence weights** on the
signal views. A signal that barely passes the falsification protocol (DSR near 0.95, $\gamma = 0.30$)
contributes its view very weakly. A well-validated signal ($\gamma = 0.95$, high DSR)
strongly tilts the portfolio. Bayesian blending prevents over-reliance on any single signal.*

### 5.4 — Transaction Cost Model & Go/No-Go Gates

$$\text{TC(bps)} = \frac{\text{spread}}{2} \cdot \Delta w \cdot 10^4 + \lambda_{\text{impact}} \cdot \sqrt{\frac{\Delta w}{\text{ADV}}}$$

Go/No-Go kill criteria (pre-agreed before live trading):

| Condition | Action |
|-----------|--------|
| $\gamma < 0.25$ | Suspend — causal validation failed |
| Turnover $> 50\%$ per session | Split across 2 sessions |
| $\text{TC} > 50\% \times$ expected gross alpha | Hold — not worth it |

---

## §6 · Six-Month Deployment Plan

[🔝 Back to Top](#-table-of-contents)

```
Month  Milestone                                          KPI Gate
─────  ─────────────────────────────────────────────────  ─────────────────────────────
1      Framework audit + LME/CME/FRED data ingestion;     Decay baseline computed
       baseline Sharpe decay analysis

2      ISCF+MGD construction + Gram-Schmidt; orthog.      |ρ| < 0.39 w/ all 3 factors
       validation (pairwise R² < 0.15)                    Max VIF < 5.0

3      CPCV (45 paths) + DSR + DoWhy causal stack;        DSR > 0.95; all 3 causal
       BH/Bonferroni FDR control                          steps pass; FDR ↓ ≥ 30%

4      Black-Litterman γ-prior integration + TCA model;   γ-weighted paper SR > 1.0
       4× daily rebalancing engine live (paper)

5      Shadow production on live vendor feeds;            Fill rate > 95%;
       execution audit (fill rate, slippage vs TCA);      Slippage < TCA model
       CI/CD cron health dashboard 4× daily

6      Production launch; health dashboard (SR decay,      Walk-forward SR > 0.70
       T½, regime alerts, γ); final FDR verification       FDR ↓ ≥ 30%
```

**KPI constants (hardcoded, not tuned post-hoc):**

```
Walk-forward SR target:    0.70     (after all haircuts)
Max baseline R²:           0.15     (orthogonality gate)
IC floor:                  0.02     (signal retirement)
ICIR floor:                0.50     (allocation reduction)
t-statistic floor:         3.0      (statistical significance)
Max single-position:       5%       (concentration limit)
Max ADV participation:     10%      (market impact cap)
```

---

## §7 · Caveats, Pitfalls & Real-World Mitigations

[🔝 Back to Top](#-table-of-contents)

### 🚨 Pitfall 1 — Proxy Data Limitations (Current Implementation)

**Problem:** The research uses ETF price proxies (USO, CPER, GLD) instead of
actual LME/CME prompt-date futures. ETFs embed management fees, tracking error,
and do not reflect actual physical delivery premiums — the core ISCF mechanism.
For MGD, FRED macro series are used with rolling-average consensus as a proxy
for Bloomberg consensus estimates.

**In practice this means:** the backtest IC is an estimate of the achievable IC, not the
true IC. The proxy-to-true IC degradation is hard to quantify without proprietary data.

**Mitigation — Day 1 data swap architecture:** The entire system is built with a
Design-by-Contract provider layer. Switching to real data requires only changing the
provider constructor:

```python
# Free tier (research):
provider = YFinanceISCFProvider()

# Production (Day 1 swap — identical interface):
provider = HLSISCFProvider(api_key=os.environ["HLS_API_KEY"])
```

Zero downstream code changes. Month 1 of the deployment plan is explicitly dedicated
to establishing the baseline IC degradation between proxy and real data.

### 🚨 Pitfall 2 — Synthetic Backtest Data

**Problem:** In the absence of full historical data, some components use synthetic
panel data calibrated to realistic statistics (IC ∈ [0.02, 0.06], vol-clustering,
autocorrelation structure). Synthetic data can preserve the right marginals while
missing the joint dynamics — particularly regime-dependent correlations.

**Mitigation:** The `backtest.ipynb` notebook runs the full pipeline on real yfinance/FRED
data (2015–2024). The CPCV protocol generates 45 out-of-sample paths, each a distinct
train/test split — so the overfitting risk is bounded by the DSR framework. Month 3's
KPI gate (DSR > 0.95 on real OOS data) is the hard validation checkpoint.

### 🚨 Pitfall 3 — Regime Blindness (2008, 2020, 2022)

**The tail events:** 2008 (liquidity crisis), 2020 (COVID), 2022 (rates regime shift).

**For ISCF:** Physical commodity markets exhibit **regime-specific basis behaviour**.
In 2008, backwardation collapsed into deep contango across energy (forced liquidations
overshadowed convenience yield). In 2022, LME Nickel had an exchange-imposed trading
halt. The signal assumes continuous, accessible markets — which fails in these scenarios.

**For MGD:** In 2020 and 2022, the signal-to-noise ratio of macro surprises collapsed
(every release was a surprise; consensus models were far off), reducing the Kalman
filter's effective window. The EMA assumed a stable $q = 0.29$ signal-to-noise ratio,
which was violated.

**Mitigations in the current framework:**
- **VIX regime gate:** VIX proxy is an exogenous control in the VARX model. High-VIX
  regimes reduce Granger significance, lowering $\gamma$ — which automatically
  reduces position sizing.
- **MAD winsorisation at $z_{\max} = 4$:** Clips the LME Nickel 2022-style 250% moves.
- **ADV cap of 10%:** In crisis liquidity drops; 10% ADV participation limit prevents
  the portfolio from becoming uninvestable.
- **Half-life monitor:** An OU half-life below 17.85 days triggers immediate signal
  retirement — regime shifts show up as accelerating IC decay before the full regime
  shift is apparent.
- **What is NOT yet in the framework:** An explicit HMM-based regime classifier
  (Hidden Markov Model) that switches signal weights between normal and crisis states.
  This is the highest-priority enhancement for Month 2–3 in the deployment plan.

### 🚨 Pitfall 4 — Multiple Testing & Overfitting

**Problem:** The two signals were selected from a larger hypothesis space. Even with
$M=2$, the Bonferroni-corrected significance level is $\alpha^* = 0.025$. Any researcher
who tested 20+ hypotheses before arriving at these two has implicitly used a much larger
$M$ — and the Bonferroni correction should reflect the full search breadth, not just the
reported signals.

**Mitigation:** The DSR explicitly penalises for $M=2$ trial configurations. The
CPCV protocol with 45 paths means there is no single heroic train/test split. The 
`constants.py` file hardcodes all thresholds *before* seeing the final results —
the equivalent of pre-registration in academic research.

### 🚨 Pitfall 5 — Transaction Cost Model Uncertainty

**Problem:** The TCA model uses estimated bid-ask spreads and an Almgren-Chriss
market impact coefficient $\lambda_{\text{impact}}$. At 4× daily rebalancing, turnover
compounds aggressively. A 5 bps underestimate in spread across 1008 sessions is a
full Sharpe point of drag annually.

**Mitigation:** Month 5 is explicitly a **fill rate audit**: every executed rebalance
is compared against the TCA model's prediction. Any systematic deviation triggers a
model recalibration. The Go/No-Go gate $\text{TC} > 50\% \times$ expected gross alpha
provides a hard circuit breaker.

### 🚨 Pitfall 6 — Gram-Schmidt Instability in Small Cross-Sections

**Problem:** Gram-Schmidt orthogonalisation is numerically sensitive to nearly
collinear factor vectors. With $N=8$ assets in the ISCF universe, the factor matrix
can become near-singular in certain regimes (e.g., crisis-driven high correlations).

**Mitigation:** VIF < 5 and $R^2 < 0.15$ are enforced as hard gates. If VIF ≥ 5,
the offending baseline factor is dropped from the orthogonalisation set for that
session. The residual signal is then marked as $\gamma = 0.30$ until VIF normalises.

---

## §8 · Data Requirements & Alternative Data Sourcing

[🔝 Back to Top](#-table-of-contents)

### ISCF — Required Data

| Data Type | Specific Series | Source (Free Tier) | Source (Production) |
|-----------|----------------|-------------------|-------------------|
| Commodity spot prices | WTI, Brent, Nat Gas, HH, Copper HG, Gold GC, Silver SI | USO/BNO/UNG/CPER/GLD/SLV ETFs via yfinance | CME Direct / LME API |
| **Prompt-date futures** | Front-month vs. deferred spreads (M1-M3, M3-M6) | Unavailable — proxy with ETF roll yield | **CME DataMine, QuikStrike, ICE Connect** |
| **Physical inventory** | LME warehouse stocks (Cu, Al, Zn, Ni); EIA crude/distillate stocks | EIA weekly (FRED series WDC...) | **LME Live Warrant Data, EIA API, Genscape** |
| Freight/logistics | Baltic Dry Index for metals logistics | Quandl/FRED (partial) | **Baltic Exchange, Clarksons Research** |
| Realised volatility | 20-day rolling from price series | Derived from above | Same, but from tick data |

### MGD — Required Data

| Data Type | Specific Series | Source (Free Tier) | Source (Production) |
|-----------|----------------|-------------------|-------------------|
| G10 FX spot | EUR, GBP, JPY, AUD, CAD vs USD | yfinance (^EURUSD, etc.) | EBS/Refinitiv Elektron |
| **FX forward rates** | 1M, 3M, 6M outrights | Unavailable — proxy with spot | **Bloomberg BFIX, Refinitiv FXall** |
| **Flash PMI** | Markit/S&P Global Manufacturing + Services Flash | ISM monthly (FRED, delayed) | **S&P Global Flash PMI (day-of release)** |
| **Consensus estimates** | Actual vs. consensus for PMI, CPI, NFP | Rolling mean proxy | **Bloomberg Economic Surprise Index, Citigroup ESI** |
| CPI / Inflation | Headline and core CPI, PPI | FRED (CPIAUCSL, CPILFESL) | Bloomberg Economics, Haver Analytics |
| Employment | NFP, AHE, unemployment rate | FRED (PAYEMS, UNRATE) | Same + ADP Research Institute |
| Central bank policy | Rate decisions, meeting minutes | FRED + Fed website | **MNI, Action Economics, Bloomberg** |

### Alternative Data — High-Value Additions

| Dataset | Signal Application | Vendor |
|---------|-------------------|--------|
| **Physical commodity flow data** (shipping manifests, port arrival) | ISCF: lead indicator for inventory changes | Kpler, Vortexa, TankerTrackers |
| **Satellite imagery** (Cushing tank levels, LME warehouse occupancy) | ISCF: real-time inventory proxy independent of official reports | Orbital Insight, SpaceKnow, Kayrros |
| **FX options implied vol surface** | MGD: regime detection; cross-asset skew for surprise magnitude scaling | Bloomberg OVDV, SuperDerivatives |
| **Real-time news sentiment** (NLP on wire releases) | MGD: pre-release directional lean from central banker speech sentiment | Refinitiv News Analytics, RavenPack |
| **Order flow imbalance** (institutional FX flow) | MGD: direct measure of the post-release institutional lag | CLS Group FX Flow Data, State Street GX |
| **Economic nowcasting indices** | MGD: replace ad-hoc PMI/CPI/NFP composite with professional nowcasts | Atlanta Fed GDPNow, NY Fed Nowcast, Goldman Sachs MAP |

---

## §9 · Quick-Reference Equation Sheet

[🔝 Back to Top](#-table-of-contents)

### Signal 1 — ISCF

| Step | Formula | Intuition |
|------|---------|-----------|
| No-arbitrage | $F(t,T) = S(t)\,e^{(r+u-c)(T-t)}$ | Futures = spot compounded at net carry |
| Vol-norm basis | $b_i = (S_i - F^{\text{def}}_i) / \max(\sigma^{rv}_i, \varepsilon)$ | Puts all commodities on same risk-adj scale |
| Robust z-score | $z_i = \text{clip}((b_i - \text{med})/(\text{MAD}+\varepsilon), -4, +4)$ | Outlier-resistant cross-section rank |
| Idio extraction | $\alpha^{\text{raw}}_i = \text{sign}(z_i)\cdot|z_i|^{1/2}\cdot(1-\beta^{\text{macro}}_i)$ | Concave + idio; strips macro beta |
| Gram-Schmidt | $\hat{\alpha}^{\text{ISCF}} = \alpha^{\text{raw}} - \sum_k \frac{\langle \alpha^{\text{raw}}, f^k\rangle}{\langle f^k, f^k\rangle}f^k$ | Residualise vs. trend/mom/carry |
| Rank normalise | $r_i = \Phi^{-1}(\text{rank}(\hat{\alpha}_i)/(N+1))$ | Map to $\mathcal{N}(0,1)$ for optimiser |

### Signal 2 — MGD

| Step | Formula | Intuition |
|------|---------|-----------|
| Surprise | $\Delta^k_i = (\text{actual}^k_i - \text{consensus}^k_i)/\sigma^k_{\text{hist}}$ | Normalised deviation from consensus |
| Composite | $\mathcal{S}_i = 0.40\Delta\text{PMI} + 0.30\Delta\text{CPI} + 0.30\Delta\text{EMP}$ | Weighted macro surprise index |
| EMA/Kalman | $\hat{\mu}_t = \alpha\mathcal{S}_t + (1-\alpha)\hat{\mu}_{t-1}$, $\alpha=2/(\tau+1)$ | Steady-state Kalman = optimal EMA |
| Divergence | $D_i = (\mathcal{S}_i - \hat{\mu}_i)/\max(\sigma^{60}_i, \varepsilon)$ | Innovation: genuinely unpriced surprise |

### Causal Framework

| Test | Formula | Pass Condition |
|------|---------|----------------|
| VARX Granger | $F = (RSS_R - RSS_U)/df_1 / \hat{\sigma}^2_{\text{HAC}}$ | $p < 0.05$ |
| HAC bandwidth | $L = \lfloor 4(T/100)^{2/9}\rfloor$ | $L \approx 12$ for $T=1000$ |
| CMI proxy | $\psi = \|\rho_{\text{partial}}\| / \|\rho_{\text{raw}}\|$ | $\psi \geq 0.50$ |
| Placebo | $p_{\text{pl}} = B^{-1}\sum \mathbf{1}(\|\hat{\theta}_b\| \geq \|\hat{\theta}_{\text{orig}}\|)$ | $p_{\text{pl}} > 0.05$ |

### Statistical Validation

| Tool | Formula | Purpose |
|------|---------|---------|
| CPCV paths | $\binom{10}{2} = 45$ | OOS distribution, no leakage |
| Bonferroni | $\alpha^* = 0.05/M$ | FWER control ($M=2 \Rightarrow \alpha^*=0.025$) |
| DSR | $\Phi\!\left((\widehat{SR}-SR^*)\sqrt{T-1}/\sqrt{1-\hat{\gamma}_3\widehat{SR}+(\hat{\gamma}_4-1)/4\cdot\widehat{SR}^2}\right)$ | Selection-bias-adjusted SR probability |
| Half-life | $T_{1/2} = \ln 2 / \hat{\kappa}$, $\Delta IC_t = -\hat{\kappa}\,IC_{t-1}$ | Signal decay clock |
| FLOAM | $\text{IR} = \text{IC}\cdot\sqrt{K\cdot\text{Breadth}_{\text{single}}}$ | Value of orthogonal breadth |

---

*Shaikat Majumdar · HLS Trading Interview Preparation · June 2026*
*ISCF & MGD: Systematic Macro Alpha from First Principles*
