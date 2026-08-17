# Properietary Trading Firm — Quantitative Alpha Research Playbook
### ISCF & MGD: Systematic Macro Alpha from First Principles
#### Two Orthogonal Signals · Causal Validation · Production-Grade Implementation

> **Delivery philosophy:** Every concept below is structured as *plain-English intuition first, mathematics as evidence second, code as proof third.* The dissertation derives both signals from economic first principles. The implementation translates every formula directly into tested, production-grade Python and C++26. This document bridges those two worlds end-to-end.

---
---

[↩️ Back to ../README.md](../README.md)

---
---

## ⏱️ Document Map

```
SECTION                                         WHAT YOU WILL UNDERSTAND
──────────────────────────────────────────────  ───────────────────────────────────────────────────────────
§0  Why Orthogonal Alpha Matters                Grinold's Fundamental Law — breadth vs. IC maths
§1  Signal 1: ISCF                              Physical inventory theory → basis → idiosyncratic extraction
§2  Signal 2: MGD                               Rational expectations → macro surprise → FX divergence
§3  Causal Validation Framework                 3-step pipeline: Granger / CMI / DoWhy → γ factor
§4  Statistical Falsification Protocol          CPCV, Deflated SR, Bonferroni/BH, half-life monitoring
§5  Portfolio Construction                      HRP + Ledoit-Wolf + Black-Litterman integration
§6  4× Daily Rebalancing Engine                 Session logic, TCA model, Go/No-Go gates
§7  Implementation Architecture                 File-by-file code walkthrough with excerpts
§8  Six-Month Deployment Plan                   Month-by-month KPIs and kill criteria
§9  Quick-Reference Equation Sheet              Every key formula in one place
```

---

## Table of Contents

- [§0 · Why Orthogonal Alpha Matters — The Fundamental Law](#0--why-orthogonal-alpha-matters--the-fundamental-law)
- [§1 · Signal 1: ISCF — Idiosyncratic Supply Chain Flow](#1--signal-1-iscf--idiosyncratic-supply-chain-flow)
  - [1.1 · Theoretical Foundation: Theory of Storage](#11--theoretical-foundation-theory-of-storage)
  - [1.2 · Volatility-Normalised Basis](#12--volatility-normalised-basis)
  - [1.3 · Robust Cross-Sectional Z-Score (MAD)](#13--robust-cross-sectional-z-score-mad)
  - [1.4 · Idiosyncratic Extraction & Macro-Beta Stripping](#14--idiosyncratic-extraction--macro-beta-stripping)
  - [1.5 · Gram-Schmidt Orthogonalisation](#15--gram-schmidt-orthogonalisation)
  - [1.6 · Gaussian Rank Normalisation](#16--gaussian-rank-normalisation)
- [§2 · Signal 2: MGD — Real-Time Macro Growth Divergence](#2--signal-2-mgd--real-time-macro-growth-divergence)
  - [2.1 · Theoretical Foundation: Rational Expectations](#21--theoretical-foundation-rational-expectations)
  - [2.2 · Composite Nowcast Surprise Index](#22--composite-nowcast-surprise-index)
  - [2.3 · EMA as Optimal Linear Predictor — Kalman Filter Derivation](#23--ema-as-optimal-linear-predictor--kalman-filter-derivation)
  - [2.4 · Divergence Signal Construction](#24--divergence-signal-construction)
- [§3 · Causal Validation Framework](#3--causal-validation-framework)
  - [3.1 · Why Prediction ≠ Causation](#31--why-prediction--causation)
  - [3.2 · Step 1: VARX Granger Causality with HAC Correction](#32--step-1-varx-granger-causality-with-hac-correction)
  - [3.3 · Step 2: Conditional Independence Test (CMI)](#33--step-2-conditional-independence-test-cmi)
  - [3.4 · Step 3: DoWhy Refutation Tests](#34--step-3-dowhy-refutation-tests)
  - [3.5 · Causal Confidence Factor γ](#35--causal-confidence-factor-γ)
- [§4 · Statistical Falsification Protocol](#4--statistical-falsification-protocol)
  - [4.1 · Combinatorial Purged Cross-Validation (CPCV)](#41--combinatorial-purged-cross-validation-cpcv)
  - [4.2 · Multiple Testing Correction](#42--multiple-testing-correction)
  - [4.3 · Deflated Sharpe Ratio](#43--deflated-sharpe-ratio)
  - [4.4 · Signal Half-Life via OU Regression](#44--signal-half-life-via-ou-regression)
- [§5 · Portfolio Construction](#5--portfolio-construction)
  - [5.1 · Ledoit-Wolf Covariance Shrinkage](#51--ledoit-wolf-covariance-shrinkage)
  - [5.2 · Hierarchical Risk Parity](#52--hierarchical-risk-parity)
  - [5.3 · Bayesian Signal Integration — Black-Litterman](#53--bayesian-signal-integration--black-litterman)
- [§6 · 4× Daily Rebalancing Engine](#6--4-daily-rebalancing-engine)
- [§7 · Implementation Architecture](#7--implementation-architecture)
- [§8 · Six-Month Deployment Plan](#8--six-month-deployment-plan)
- [§9 · Quick-Reference Equation Sheet](#9--quick-reference-equation-sheet)

[🔝 Back to Top](#table-of-contents)

---
---

# §0 · Why Orthogonal Alpha Matters — The Fundamental Law

**Open with the intuition (15 seconds):**
> "Information Ratio grows with the *square root* of the number of *independent* bets. Adding a second signal that is 95% correlated with your existing trend signal contributes almost nothing. Adding a signal that is genuinely orthogonal to trend, momentum, and carry — exploiting a completely different physical mechanism — multiplies your effective breadth. ISCF and MGD are designed to be those orthogonal signals."

---

### Grinold's Fundamental Law

$$\text{IR} = \text{IC} \cdot \sqrt{\text{Breadth}}$$

where:
- **IC** = Information Coefficient = $\mathbb{E}[\text{corr}(\hat{r}_i, r_i)]$ — the average rank correlation between your signal and forward returns
- **Breadth** = number of *independent* bets per year

```
FUNDAMENTAL LAW — INTUITION

  IC = 0.05 (very good for macro)
  Breadth = 1,008 (one daily signal across N assets, 4x/day)

  IR = 0.05 * sqrt(1008) = 0.05 * 31.7 = 1.59

  Add a second CORRELATED signal (ρ = 0.80):
    Effective breadth barely changes: IR ≈ 1.65   (+4%)

  Add a second ORTHOGONAL signal (ρ ≈ 0):
    Breadth doubles: IR = 0.05 * sqrt(2016) = 2.24  (+41%)

  This is why orthogonality is not a nice-to-have — it is the maths.
```

At Properietary Trading Firm's 4× daily frequency across N assets:

$$\text{Breadth}_\text{HLS} = 4 \times 252 \times N_\text{assets}$$

Adding $K$ mutually orthogonal signals scales the IR as:

$$\text{IR}_\text{total} = \text{IC} \cdot \sqrt{K \cdot \text{Breadth}_\text{single}} = \sqrt{K} \cdot \text{IR}_\text{single}$$

### What the Existing Signals Already Capture

```
SIGNAL SPACE AUDIT

  Factor           Definition                           What it captures
  ───────────────  ───────────────────────────────────  ────────────────────────────────────────────
  Trend            sign(r_{t-252:t-21})                 Long-run price momentum (12-1 month)
  Momentum         r_{t-21:t}                           Short-term price continuation/reversal
  Carry            r_d - r_f (interest rate diff.)      Yield advantage of holding an asset

  UNSPANNED DIMENSIONS:
  → Physical delivery constraints in futures markets   ← ISCF targets this
  → Real-time macro data arrival & market reaction lag ← MGD targets this
```

**The practical constraint:** signals must be *genuinely* independent — not just different parameterisations of the same latent factor. Both ISCF and MGD are residualised against the baseline three via Gram-Schmidt orthogonalisation before they receive any capital allocation.

---

[🔝 Back to Top](#table-of-contents)

---
---

# §1 · Signal 1: ISCF — Idiosyncratic Supply Chain Flow

**Open with the intuition (15 seconds):**
> "When physical commodity inventory falls below a critical threshold, holders of the physical commodity gain an enormous convenience yield — the market pays them a premium to hold the actual metal or barrel today rather than lock it up in a warehouse for future delivery. This drives the futures curve into steep backwardation. ISCF reads this signal from the vol-normalised basis, strips out the market-wide macro beta, and residualises against trend/momentum/carry to isolate the genuinely idiosyncratic physical-scarcity premium."

---

## 1.1 · Theoretical Foundation: Theory of Storage

Working (1949) and Fama & French (1988) establish the no-arbitrage condition for storable commodities:

$$F(t, T) = S(t) \cdot e^{(r + u - c)(T - t)}$$

where:
- $S(t)$ = spot price today
- $F(t, T)$ = futures price for delivery at time $T$
- $r$ = risk-free rate
- $u$ = storage cost (warehousing, insurance)
- $c$ = **convenience yield** — the flow of services from holding the *physical* commodity

The convenience yield is a non-linear, decreasing function of inventory $I_t$:

$$c = g(I_t) = \begin{cases} c_{\min} & I_t > I^{\*} \\ g_0 + g_1/I_t & I_t \leq I^{\*} \end{cases}$$

```
CONVENIENCE YIELD vs INVENTORY — THE HOCKEY STICK

  c (convenience
  yield)
     │
 c_max│──────────────────────────────\
     │                                \
     │                                 \
     │                                  \
 c_min│                                   ─────────────────────────
     │
     └──────────────────────────────────────────────────────────────
       0              I*                                   I_t
                  (critical               (abundant inventory)
                  threshold)

  Below I*: convenience yield explodes → curve into steep backwardation
  Above I*: convenience yield at floor  → curve in contango or flat
```

**Why this creates a tradeable alpha:**
When $I_t < I^{\*}$, the convenience yield dominates storage costs: $F \ll S$ (backwardation). The physical scarcity premium is priced into the curve, but:
1. Institutional constraints prevent immediate physical arbitrage (delivery logistics, storage capacity)
2. Slow-moving commercial hedgers mean the adjustment completes over days, not seconds
3. This creates a *predictable* return series within HLS's 4× daily holding window

---

## 1.2 · Volatility-Normalised Basis

**Intuition:** "A $2/bbl basis between WTI front and deferred is trivial noise when WTI vol is 45%. The same $2/bbl basis on a low-vol metal is a screaming signal. We need to put every commodity on a common footing before comparing them cross-sectionally."

> [!TIP]
>
> In the energy and commodities markets, **bbl** is the standard industry abbreviation for a **barrel of crude oil**.
>
> One barrel (1 bbl) is exactly equal to **42 U.S. gallons** (or approximately 159 liters).
>
> So, when you see a phrase like *"A $2/bbl basis between WTI front and deferred,"* it translates to a $2.00 difference per barrel between the Western Texas Intermediate (WTI) near-month futures contract ("front") and a later-expiring futures contract ("deferred").
>
> ---
> ### Why "bbl" and not just "bl"?
>
> The origin of the double "b" is a bit of historical lore from the early days of the Pennsylvania oil boom in the 1860s:
>
> * **Standard Oil's Blue Barrels:** The most widely accepted explanation is that Standard Oil (founded by John D. Rockefeller) began manufacturing its shipping barrels out of wood painted bright blue to guarantee buyers that they were getting a full, standard 42-gallon measure. The abbreviation "bbl" stood for **"blue barrel."**
> * **Pluralization:** A secondary theory is that it was simply used as an old-style abbreviation to denote the plural (e.g., *bl.* for barrel, *bbl.* for barrels), which eventually just stuck as the permanent unit symbol.
>
> Today, whether you are dealing with physical crude or trading financial derivatives on NYMEX, **bbl** is the universal shorthand.

For asset $i$ at time $t$, define the volatility-normalised basis:

$$b_{i,t} = \frac{S_{i,t} - F^{\text{def}}_{i,t}}{\max(\sigma^{rv}_{i,t}, \ \varepsilon)}$$

where the realised volatility is computed over a 20-day rolling window:

$$\sigma^{rv}_{i,t} = \sqrt{\frac{252}{H} \sum_{h=1}^{H} \left(\log \frac{S_{i,t-h+1}}{S_{i,t-h}}\right)^2}, \quad H=20, \quad \varepsilon=10^{-8}$$

```
BASIS NORMALISATION — WORKED EXAMPLE

  Asset         Spot     Deferred  Raw Basis  RVol     Normalised Basis
  ────────────  ───────  ────────  ─────────  ───────  ────────────────
  Copper (HG)   $9,200   $8,800    +$400      18%      +2.22 σ
  Nat Gas (NG)  $2.50    $2.30     +$0.20     45%      +0.44 σ
  Gold (GC)     $2,100   $2,090    +$10       12%      +0.08 σ

  Without normalisation: Copper looks like a small absolute move.
  After normalisation:   Copper is a 2.2-sigma event. That IS the signal.
```

**Code implementation** — `signals_hls.py`:
```python
def _compute_basis_zscore(spot, deferred, rvol):
    """Compute volatility-normalised basis z-score (cross-sectional)."""
    eps = 1e-8
    basis = (spot - deferred) / np.maximum(rvol, eps)   # vol-normalised basis
    med = float(np.median(basis))
    mad = float(np.median(np.abs(basis - med)))
    z = (basis - med) / max(mad, eps)                   # MAD robust z-score
    return np.clip(z, -C.ISCF_MAX_BASIS_ZSCORE, C.ISCF_MAX_BASIS_ZSCORE)
```

The winsorisation cap `ISCF_MAX_BASIS_ZSCORE = 4.0` (from `constants.py`) corresponds to probability mass $< 3.2 \times 10^{-5}$ under Gaussianity — protecting against the LME Nickel 2022 and WTI April 2020 squeeze events.

---

## 1.3 · Robust Cross-Sectional Z-Score (MAD)

**Intuition:** "Commodity markets have violent outliers — LME Nickel moved 250% in two days in 2022. A classical mean-based z-score would be completely destroyed by such an event. The MAD estimator has a 50% breakdown point: you can contaminate *half your sample* with extreme values and the estimator still converges to the true location. That is the correct tool for commodity cross-sections."

The robust z-score uses the **Median Absolute Deviation**:

$$z_{i,t} = \text{clip}\left(\frac{b_{i,t} - \text{median}_j(b_{j,t})}{\text{MAD}_j(b_{j,t}) + \varepsilon},\; -z_{\max},\; z_{\max}\right)$$

$$\text{MAD}_j = \text{median}_j\left|b_j - \text{median}(b)\right|$$

```
ROBUSTNESS COMPARISON: MEAN vs MAD

  Classical z-score:                  MAD z-score:
  ─────────────────────────────────── ─────────────────────────────────────
  mean(basis) = 0.10 (distorted       median(basis) = 0.05 (stable)
                by outlier)
  std(basis)  = 2.50 (inflated)       MAD(basis)    = 0.08 (stable)

  Result: all assets look tame,        Result: outlier gets winsorised,
          signal swamped by noise.             genuine signals preserved.

  Breakdown point: 0% (one extreme    Breakdown point: 50% (half the
  outlier corrupts the estimator)     sample can be corrupt)
```

**Lemma (Robustness of MAD):** The influence function of the MAD estimator satisfies $\sup_b |IF(b; \text{MAD}, F)| < \infty$, confirming bounded sensitivity to any single extreme observation.

This is equivalent to a **one-step M-estimator** with Huber loss $\rho(x) = \min(x^2, c|x| - c^2/2)$ for $c=1.345$ — the theoretically optimal choice achieving 50% breakdown point while retaining 95% efficiency under Gaussian conditions.

---

## 1.4 · Idiosyncratic Extraction & Macro-Beta Stripping

**Intuition:** "If Copper drops because *everything* risk-off dropped, that is not an ISCF signal — it is macro beta. We want the part of the basis that is specific to Copper's physical supply chain, net of the broad commodity-as-risk-asset exposure. The $(1-\beta^\text{macro})$ factor does exactly that."

$$\alpha^{\text{ISCF,raw}}_{i,t} = \underbrace{\text{sign}(z_{i,t}) \cdot |z_{i,t}|^{1/2}}_{\text{concave signal response}} \cdot \underbrace{(1 - \beta^\text{macro}_{i,t})}_{\text{idiosyncratic mask}}$$

The **square-root dampening** $|\cdot|^{1/2}$ serves two purposes:
1. Reduces excess kurtosis of the signal distribution from ~6 to ~2, improving portfolio-optimiser stability
2. Implements a concave utility-like response — the marginal expected return contribution diminishes as the signal becomes extreme (consistent with *limits to arbitrage*: large basis divergences are harder to trade at scale)

The **macro-beta** $\beta^\text{macro}_{i,t}$ is estimated via 60-day rolling OLS of asset returns against a broad commodity index:

```
MACRO-BETA STRIPPING — INTUITION

  Raw basis signal:
  α_raw = +2.1σ for Copper (strong backwardation)

  But β_macro = 0.75 (Copper moves 75% with broad risk-off)

  Idiosyncratic component:
  α_ISCF = sign(2.1) * sqrt(2.1) * (1 - 0.75)
          = +1.449 * 0.25
          = +0.36

  Interpretation: After removing the broad macro move, the genuine
  supply-chain signal is +0.36 — modest but genuinely idiosyncratic.

  An asset with β_macro = 0.05 and same z-score:
  α_ISCF = +1.449 * 0.95 = +1.38  ← much stronger
```

**Code** — `signals_hls.py`, `compute_iscf()`:
```python
beta = np.clip(np.asarray(macro_beta, dtype=np.float64), 0.0, 1.0)
# Idiosyncratic component: dampen by macro beta exposure
raw = np.sign(basis_z) * np.sqrt(np.abs(basis_z)) * (1.0 - beta)
```

---

## 1.5 · Gram-Schmidt Orthogonalisation

**Intuition:** "Gram-Schmidt is the mathematical equivalent of asking: *what does this signal know, that trend, momentum, and carry don't already know?* You project out the components of ISCF that lie along the existing factor directions, leaving only the genuinely new information."

Let $F_t = [f^{\text{trend}}_t, f^{\text{mom}}_t, f^{\text{carry}}_t] \in \mathbb{R}^{N \times 3}$. The residualised signal is:

$$\hat{\alpha}^{\text{ISCF}}_t = \alpha^{\text{ISCF,raw}}_t - \sum_{k=1}^{3} \frac{\langle \alpha^{\text{ISCF,raw}}_t, \ f^k_t \rangle}{\langle f^k_t, \ f^k_t \rangle} f^k_t$$

```
GRAM-SCHMIDT — GEOMETRIC INTUITION

  Think of signals as vectors in N-dimensional space:

        f_carry
            ↑
            │   α_ISCF_raw
            │  ╱
            │ ╱   ← projection onto carry
            │╱────────────────── f_trend
            └──────────────────────────────
           origin

  After Gram-Schmidt:
  α_hat_ISCF = α_ISCF_raw - proj_trend - proj_mom - proj_carry

  This vector is EXACTLY PERPENDICULAR to all three baseline factors.
  It contains ONLY the information not already in the baseline space.
```

**Proposition:**

$$
\langle \hat{\alpha}^{\text{ISCF}}_t, f^k_t \rangle = 0 \quad \text{for all } k \in \{1,2,3\}
$$

The Gram-Schmidt projection is the unique minimum-norm solution to:

$$
\min_{\hat{\alpha}} \|\hat{\alpha} - \alpha^{\text{raw}}\|_2 \quad \text{subject to } F_t^T \hat{\alpha} = 0
$$

**Production gate (KPI Month 2):** `analytics.py` verifies $R^2 < 0.15$ between the orthogonalised signal and every baseline factor before the signal is allocated capital:

```python
# analytics.py — compute_orthogonality()
r2_matrix = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        rho, _ = spearmanr(signal_matrix[:, i], signal_matrix[:, j])
        r2_matrix[i, j] = float(rho ** 2)
# Gate: max off-diagonal R^2 < MAX_R2_ORTHOGONALITY = 0.15
is_orthogonal = max_r2 < threshold_r2 and bool(np.all(vif < threshold_vif))
```

**Code** — `signals_hls.py`, `gram_schmidt_residualise()`:
```python
def gram_schmidt_residualise(signal, baselines):
    s = signal.copy().astype(np.float64)
    for k in range(baselines.shape[1]):
        bk = baselines[:, k]
        denom = float(np.dot(bk, bk))
        if denom < 1e-12:
            continue
        proj = float(np.dot(s, bk)) / denom
        s = s - proj * bk        # Remove component along each baseline
    return s
```

> [!IMPORTANT]
> In the above context, $R^2$ (the **Coefficient of Determination**) serves as a strict mathematical constraint used to enforce **orthogonality and risk-factor neutralization**.
>
> When a quantitative system evaluates a new trading alpha, it must ensure that the signal is genuinely capturing unique anomalies rather than simply repackaging known risk exposures (such as market beta, momentum, value, or macro risk factors). Setting a production gate of $R^2 < 0.15$ means that **no more than 15% of the variance in your signal can be explained by any single baseline factor.**
>
> Here is the deep mathematical breakdown of what is happening under the hood in `analytics.py`.
>
> ---
>
> ## 1. The Mathematical Framework
>
> Let $S \in \mathbb{R}^N$ be your orthogonalized alpha signal vector across $N$ observations (or cross-sectional assets), and let $F_k \in \mathbb{R}^N$ be the vector corresponding to the $k$-th baseline factor.
>
> To calculate $R^2$, `analytics.py` runs a ordinary least squares (OLS) linear regression of your signal $S$ against the baseline factor $F_k$:
>
> $$S = \alpha + \beta_k F_k + \epsilon$$
>
> Where:
>
> * $\alpha$ is the intercept.
> * $\beta_k$ is the sensitivity (loading) of your signal to the factor $F_k$.
> * $\epsilon \in \mathbb{R}^N$ is the residual error vector.
>
> ### The Variance Decomposition
>
> By the principle of Ordinary Least Squares, the total sum of squares ($SS_{\text{tot}}$) of the signal can be decomposed into the sum of squares of the residuals ($SS_{\text{res}}$) and the explained (regression) sum of squares ($SS_{\text{reg}}$).
>
> Let $\bar{S}$ be the mean of the signal vector: $\bar{S} = \frac{1}{N}\sum_{i=1}^N S_i$.
>
> * **Total Sum of Squares ($SS_{\text{tot}}$):** Measures the total variance of your signal.
>
> $$SS_{\text{tot}} = \sum_{i=1}^N (S_i - \bar{S})^2$$
>
> * **Residual Sum of Squares ($SS_{\text{res}}$):** Measures the unexplained variance after projecting onto the factor.
>
> $$SS_{\text{res}} = \sum_{i=1}^N (S_i - (\alpha + \beta_k F_{k,i}))^2 = \sum_{i=1}^N \epsilon_i^2$$
>
> ---
>
> ## 2. Defining $R^2$ Formally
>
> The coefficient of determination is defined as:
>
> $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$
>
> Alternatively, because this is a simple linear regression (one independent variable $F_k$), **$R^2$ is exactly equal to the square of the Pearson correlation coefficient ($\rho$)** between the signal and the factor:
>
> $$R^2 = \rho^2_{S, F_k} = \left( \frac{\text{Cov}(S, F_k)}{\sigma_S \sigma_{F_k}} \right)^2$$
>
> Where:
>
> * $\text{Cov}(S, F_k) = \frac{1}{N-1}\sum_{i=1}^N (S_i - \bar{S})(F_{k,i} - \bar{F_k})$
> * $\sigma_S$ and $\sigma_{F_k}$ are the standard deviations of the signal and factor, respectively.
>
> ---
>
> ## 3. Interpreting the Gate: $R^2 < 0.15$
>
> If the gate mandates $R^2 < 0.15$, we can back out the maximum allowable correlation between your signal and any factor:
>
> $$\rho^2_{S, F_k} < 0.15 \implies |\rho_{S, F_k}| < \sqrt{0.15} \approx 0.3873$$
>
> This means that `analytics.py` will reject the signal if its absolute correlation with *any* baseline factor is **$\ge 0.39$**.
>
> ### Why is this called "orthogonalised" if $R^2$ isn't exactly 0?
>
> In an ideal world, an orthogonalization step (such as a Gram-Schmidt process, Frisch-Waugh-Lovell projection, or a symmetric Loewdin transformation) is applied ex-ante to strip out factor exposures:
>
> $$S_{\text{orthogonal}} = S_{\text{raw}} - F(F^T F)^{-1}F^T S_{\text{raw}}$$
>
> However, this production gate acts as a **failsafe verification check**. If the ex-ante orthogonalization worked perfectly over the in-sample period, $R^2$ should be close to 0.
>
> If `analytics.py` catches an $R^2 \ge 0.15$ out-of-sample or in a backtest validation phase, it implies:
>
> 1. **Regime Shift / Non-linearities:** The relationships between the factors have shifted, causing a breakdown in the linear projection.
> 2. **Look-ahead Bias / Leakage:** Factor exposure leaked back into the signal generation process.
> 3. **Spurious Correlation:** Out-of-sample data is introducing severe multi-collinearity that wasn't accounted for in the static covariance matrix used during the initial orthogonalization.
>
> By capping $R^2$ at 0.15, the portfolio management platform ensures that the capital allocated to this strategy is buying **idiosyncratic alpha** rather than paying active fees for passive factor betas.

---

## 1.6 · Gaussian Rank Normalisation

**Intuition:** "The portfolio optimiser (HRP with Ledoit-Wolf) assumes elliptical return distributions. We can force the marginal distribution of our signal to be exactly $\mathcal{N}(0,1)$ by mapping ranks through the normal quantile function. Outliers become irrelevant — only their rank matters."

$$r_{i,t} = \Phi^{-1}\!\left(\frac{\text{rank}(\hat{\alpha}^{\text{ISCF}}_{i,t})}{N+1}\right)$$

This transformation guarantees:
1. Marginal distribution $\mathcal{N}(0,1)$ by construction
2. Outlier resistance — only ordinal rank matters, not magnitude
3. Compatibility with elliptical-distribution assumptions in HRP

**Code** — `signals.py`, `_gaussian_rank_normalize()`:
```python
def _gaussian_rank_normalize(x):
    n = len(x)
    order = np.argsort(x)
    rank = np.empty(n)
    for r, idx in enumerate(order):
        u = (r + 1.0) / (n + 1.0)          # Uniform (0,1) spacing
        rank[idx] = float(stats.norm.ppf(u)) # Φ^{-1}(u) → N(0,1)
    return rank
```

The C++ engine implements the same via Blom's rational approximation to $\Phi^{-1}$ for hot-path performance (`alpha_engine.cpp`, `GaussianRankNormalize()`).

---

### Complete ISCF Pipeline Diagram

```
RAW DATA                 SIGNAL CONSTRUCTION              PORTFOLIO INPUT
─────────────────────    ───────────────────────────────  ──────────────────────
spot[i,t]            →  basis_i = (spot - deferred)      →  z_i (MAD robust)
deferred[i,t]            / max(rvol, ε)
rvol[i,t] (20d RVol) →                                   →  idio_i = sign(z)
                                                              * sqrt(|z|)
macro_beta[i,t]      →  * (1 - β_macro)                  →  orth_i (Gram-Schmidt
(60d OLS vs index)                                            residual)
                                                          →  rank_i = Φ⁻¹(rank/N+1)
baseline factors     →  - Σ proj(raw, f_k) * f_k          →  ISCF signal ∈ N(0,1)
[trend,mom,carry]

      ↓                        ↓                                ↓
IC check (≥0.02)     Causal validation (γ)            HRP weight * γ
```

[🔝 Back to Top](#table-of-contents)

---
---

# §2 · Signal 2: MGD — Real-Time Macro Growth Divergence

**Open with the intuition (15 seconds):**
> "If the market expects 50 in PMI and gets 55, something fundamental just shifted. The forward FX curve was priced for 50. The rational thing is for the currency to move immediately to reflect 55. In practice, institutional flow lags, risk constraints, and index rebalancing mean the full adjustment takes 1–6 hours. At HLS's 4× daily cadence, this adjustment window is *exactly* one holding period. MGD reads the surprise and trades the incomplete adjustment."

---

## 2.1 · Theoretical Foundation: Rational Expectations

**Muth (1961):** Under rational expectations, priced-in forward-curve expectations are *unbiased*:

$$\mathbb{E}_t[S_{t+h}] = F_{t,h}$$

Any systematic divergence between realised macro outcomes and priced-in forecasts must be attributable to:
1. **New information arrival** (unexpected macro data release)
2. **Limits to arbitrage** preventing immediate price adjustment (institutional constraints, risk limits, index flows)

**Andersen et al. (2003)** provide the empirical foundation:
- Macro data releases (NFP, CPI, PMI) generate significant FX price responses **within 5 minutes** of publication
- Partial adjustment completes over **1–6 hours**
- At Properietary Trading Firm's 4× daily rebalancing cadence, this 1–6 hour window falls squarely within a holding period

```
MACRO ANNOUNCEMENT ADJUSTMENT TIMELINE

  t=0        t=5min      t=30min     t=2h        t=6h        t=1d
  ──────────  ──────────  ──────────  ──────────  ──────────  ──────────
  Release    │Initial    │           │           │Full adj.  │
  drops      │spike      │           │           │complete   │
             │  20-40%   │  40-60%   │  60-80%   │  ~100%    │
             │  of move  │  of move  │  of move  │  of move  │

  Properietary Trading Firm 4x rebalancing sessions:
  ╔═══════════╗  ╔════════╗  ╔═════════╗  ╔══════════╗
  ║  ASIA     ║  ║LONDON  ║  ║ NY OPEN ║  ║ NY CLOSE ║
  ╚═══════════╝  ╚════════╝  ╚═════════╝  ╚══════════╝

  MGD CAPTURES: the gap between t=5min and t=6h
  (institutional adjustment lag window)
```

---

## 2.2 · Composite Nowcast Surprise Index

For FX pair $i$ at time $t$:

$$\mathcal{S}_{i,t} = w^{\text{PMI}} \Delta\text{PMI}_{i,t} + w^{\text{CPI}} \Delta\text{CPI}_{i,t} + w^{\text{EMP}} \Delta\text{EMP}_{i,t}$$

with weights $(w^{\text{PMI}}, w^{\text{CPI}}, w^{\text{EMP}}) = (0.40, 0.30, 0.30)$, $\sum w^k = 1$.

Each component is a standardised surprise:
$$\Delta^k_{i,t} = \frac{\text{actual}^k_{i,t} - \text{consensus}^k_{i,t}}{\sigma^k_{\text{hist},i}}$$

```
WEIGHT RATIONALE — WHY PMI > CPI ≈ EMP

  Post-2010 anchored-inflation environment (Andersen et al. 2003):

  Release Type     FX Move (5-min)   Persistence (6h)   Weight
  ───────────────  ────────────────  ─────────────────  ──────
  Flash PMI        Large (activity   High (growth        0.40
  (manufacturing)  forward-looking)  regime signal)
  CPI              Medium            Medium              0.30
  (anchored infl.) (rate path)       (CB reaction fn.)
  NFP/Employment   Medium            Medium              0.30
                   (labor market)    (Fed hawkishness)

  In pre-2008 environment, CPI weight was higher.
  The 40/30/30 split reflects empirical calibration for 2010–2024.
```

**Code** — `signals_hls.py`, `_composite_surprise()`:
```python
def _composite_surprise(pmi_surprise, cpi_surprise, emp_surprise):
    return (
        C.MGD_PMI_WEIGHT    * np.asarray(pmi_surprise)   # 0.40
        + C.MGD_INFLATION_WEIGHT * np.asarray(cpi_surprise)  # 0.30
        + C.MGD_EMPLOYMENT_WEIGHT * np.asarray(emp_surprise) # 0.30
    )
```

---

## 2.3 · EMA as Optimal Linear Predictor — Kalman Filter Derivation

**Intuition:** "The forward-curve-priced expectation is a weighted average of everything the market has already processed. We model it as a Kalman filter with a local-level state, and prove that the steady-state optimal filter is exactly an EMA. This is not heuristic — it is the mathematically optimal linear predictor."

Define the forward-curve-priced expectation $\mu_t = \mathbb{E}^{\text{FWD}}_t[\mathcal{S}]$ as the steady-state Kalman filter estimate for:

$$\mathcal{S}_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon) \quad \text{(observation noise)}$$
$$\mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2_\eta) \quad \text{(state evolution)}$$

Under stationarity, the Kalman gain converges to:

$$
k^{\*} = \alpha = \frac{\sqrt{q^2 + 4q} - q}{2}, \quad q = \frac{\sigma^2_\eta}{\sigma^2_\varepsilon}
$$

The steady-state filter is an EMA:

$$
\hat{\mu}_t = \alpha \mathcal{S}_t + (1 - \alpha)\hat{\mu}_{t-1}, \quad \alpha = \frac{2}{\tau_{\text{EMA}} + 1}, \quad \tau_{\text{EMA}} = 5 \text{ days}
$$

```
KALMAN ↔ EMA CORRESPONDENCE

  Kalman Filter (optimal for linear Gaussian state-space):
    predict:  μ_t|t-1 = μ_t-1
    update:   μ_t = μ_t-1 + k*(S_t - μ_t-1)
              where k = σ²_η / (σ²_η + σ²_ε)  [Kalman gain]

  EMA:
    μ_t = α * S_t + (1-α) * μ_t-1

  These are IDENTICAL when α = k* = steady-state Kalman gain.
  τ_EMA = 5 days ↔ q ≈ 0.29 (signal-to-noise ratio)
  This gives the minimum MSE linear estimate of μ_t.
```

**Code** — `signals_hls.py`, `compute_mgd()`:
```python
alpha_ema = 2.0 / (C.MGD_SURPRISE_EMA_SPAN + 1.0)  # τ=5 → α=1/3
smoothed = np.zeros(n)
smoothed[0] = divergence[0]
for i in range(1, n):
    smoothed[i] = alpha_ema * divergence[i] + (1.0 - alpha_ema) * smoothed[i - 1]
```

---

## 2.4 · Divergence Signal Construction

$$D_{i,t} = \frac{\mathcal{S}_{i,t} - \hat{\mu}_{i,t}}{\max(\sigma^{60}_{i,t}, \ \varepsilon)}$$

The numerator:

$$
\mathcal{S}_{i,t} - \hat{\mu}_{i,t}
$$

is the **Kalman filter innovation** — the unexpected component of the macro release net of all prior information.

Under $H_0$ (efficient markets): $\mathbb{E}[D_{i,t}] = 0$.

Under $H_1$ (limits to arbitrage): $\mathbb{E}[D_{i,t}] \neq 0$ in the short run, and this predictability persists for 1–6 hours.

```
MGD SIGNAL EXAMPLE — EUR/USD

  Prior EMA expectation (μ):   Flash PMI composite = +0.15 (mild positive)
  Actual composite release:    S = +0.65 (strong positive surprise)
  Rolling 60d std (σ_60):      0.20

  Innovation = S - μ = 0.65 - 0.15 = +0.50
  D = 0.50 / 0.20 = +2.5σ

  After Gram-Schmidt: removes carry and trend components
  After Gaussian rank: maps to Φ⁻¹(rank/N+1) ∈ N(0,1)

  Signal: BUY EUR/USD (or currency of positive PMI surprise)
  Holding: until next rebalancing session (~6h)
  Expected edge: incomplete price adjustment in institutional flows
```

The signal then follows the same Gram-Schmidt orthogonalisation (§1.5) and Gaussian rank normalisation (§1.6) pipeline as ISCF.

**Data architecture** — `data_provider.py`, `YFinanceMGDProvider`:

The free-tier implementation uses FRED macro series proxied via `pandas-datareader` with rolling-average consensus as a proxy for Bloomberg consensus. The `HLSMGDProvider` stub — identical `AbstractMGDProvider` interface — is the Day 1 plug-in replacement using Bloomberg consensus PMI/CPI/NFP feeds. Zero code changes outside the provider constructor.

```python
# data_provider.py — Design-by-Contract plug-in swap
# Free tier:
provider = YFinanceMGDProvider()   # FRED proxies

# Properietary Trading Firm (Day 1 swap — SAME INTERFACE):
provider = HLSMGDProvider(api_key=os.environ["HLS_API_KEY"])

# All downstream code unchanged:
data = provider.fetch(start="2015-01-01", end="2024-12-31")
```

[🔝 Back to Top](#table-of-contents)

---
---

# §3 · Causal Validation Framework

**Open with the intuition (15 seconds):**
> "A signal can be predictive in-sample for three bad reasons: it is correlated with a latent confound (spurious), the microstructure regime has shifted (structural break), or the researcher data-mined it (selection bias). The causal stack is a three-step filter that kills signals for all three reasons before they get near capital. Only a signal with a genuine economic mechanism, orthogonal to known confounds, and stable across regimes gets γ = 0.95."

---

## 3.1 · Why Prediction ≠ Causation

```
SIGNAL MORTALITY — THREE FAILURE MODES

  Failure Mode 1: SPURIOUS CORRELATION
    Signal and returns both driven by latent Z_t (e.g., VIX risk-off)
    Fix: Granger VARX with Z_t as exogenous control
    Symptom: signal drops dead when VIX regime shifts

  Failure Mode 2: MICROSTRUCTURE SHIFT
    Exchange rule change, algo update, liquidity regime shift
    Fix: Policy invariance test (structural stability across regimes)
    Symptom: signal works pre-2020, stops working post-2020

  Failure Mode 3: DATA SNOOPING
    In-sample optimisation masquerading as genuine alpha
    Fix: Placebo test (replace signal with noise, test significance)
    Symptom: in-sample SR = 2.5, out-of-sample SR = -0.1
```

---

## 3.2 · Step 1: VARX Granger Causality with HAC Correction

**Intuition:** "Granger causality asks: *does the signal predict returns over and above everything else I already know?* The VARX model controls for lagged returns AND exogenous confounders (session effects, VIX). The HAC correction prevents false rejections from autocorrelated errors giving spuriously narrow confidence intervals."

The VARX model:
$$Y_t = \sum_{k=1}^{p} A_k Y_{t-k} + \sum_{k=1}^{p} B_k X_{t-k} + C Z_t + \varepsilon_t$$

$H_0: B_1 = \cdots = B_p = 0$ (signal has no independent causal content above the exogenous controls).

The HAC-adjusted F-statistic:
$$F = \frac{(RSS_R - RSS_U) / df_1}{\hat{\sigma}^2_{\text{HAC}}}$$

The **Newey-West HAC variance** with Andrews (1991) optimal bandwidth:

$$
\hat{\Sigma}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_{j=1}^{L} \left(1 - \frac{j}{L+1}\right)\left(\hat{\Gamma}_j + \hat{\Gamma}_j^T\right), \quad L = \left\lfloor 4 \left(\frac{T}{100}\right)^{2/9} \right\rfloor
$$

```
WHY HAC MATTERS — MACRO TIME SERIES

  Without HAC:                        With HAC:
  ─────────────────────────────────── ─────────────────────────────────────
  OLS assumes εt ~ iid                Macro returns are autocorrelated
  SE too small by factor of ~2-3×     SE correctly inflated
  p-value = 0.01 (spurious pass)      p-value = 0.08 (correct fail)

  Example: VIX term structure has AR(5) residuals.
  Without HAC: every macro signal "Granger-causes" VIX moves.
  With HAC L=12: only genuine lead-lag relationships survive.
```

**Code** — `causal.py`, `granger_causality_varx()`:
```python
def granger_causality_varx(signal, returns, exogenous=None, max_lags=2, alpha=0.05):
    for lag in range(1, max_lags + 1):
        # Restricted: Y ~ Y_lags [+ Z]; Unrestricted: + X_lags
        ...
        e_u = y_dep - x_u @ beta_u
        hac_var = _newey_west_variance(e_u)           # HAC correction
        f_stat = ((rss_r - rss_u) / df1) / max(hac_var, 1e-12)
        p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))
```

```python
def _newey_west_variance(residuals, lag_truncation=12):
    """Σ_HAC = Γ_0 + Σ w_j (Γ_j + Γ_j^T), Bartlett kernel."""
    gamma0 = float(np.dot(e, e) / t)
    hac = gamma0
    for j in range(1, lag_truncation + 1):
        weight = 1.0 - j / (lag_truncation + 1.0)
        gamma_j = float(np.dot(e[j:], e[:-j]) / t)
        hac += 2.0 * weight * gamma_j
    return max(hac, 1e-12)
```

**Pass condition:** $p < 0.05$ (reject $H_0$ that signal adds no information above exogenous controls).

---

## 3.3 · Step 2: Conditional Independence Test (CMI)

**Intuition:** "Even if the signal Granger-causes returns, it might be doing so primarily *through* known risk factors — it might be 80% beta-proxy and only 20% genuine alpha. The CMI test measures what fraction of the raw predictive content survives after conditioning on all known confounders. Below 50% → route to beta-proxy basket, not independent capital."

The conditional mutual information proxy uses partial correlation:

$$I(X; Y \mid Z) \approx \frac{1}{2} \ln \frac{1}{1 - \rho^2_{\text{partial}}}$$

where $\rho_{\text{partial}} = \text{corr}(\tilde{X}, \tilde{Y})$ and $\tilde{X} = X - \hat{X}(Z)$, $\tilde{Y} = Y - \hat{Y}(Z)$ are OLS residuals.

The **alpha retained fraction**:
$$\psi = \frac{|\rho_{\text{partial}}|}{|\rho_{\text{raw}}|}$$

```
CMI TEST — DECISION TREE

  ψ ≥ 0.50 (≥50% alpha retained after conditioning):
    → Signal is genuinely idiosyncratic
    → Proceed to Step 3 (DoWhy)

  ψ < 0.50 (signal is >50% beta to known confounders):
    → Route to "beta-proxy" basket
    → γ = 0.30 (reduced capital)
    → Do NOT count toward orthogonal breadth expansion
```

**Code** — `causal.py`, `conditional_independence_test()`:
```python
def conditional_independence_test(signal, returns, confounders, alpha=0.05):
    # Residualise signal and returns on confounders
    x_res = _residualise(signal)     # X̃ = X - X̂(Z)
    y_res = _residualise(returns)    # Ỹ = Y - Ŷ(Z)

    raw_corr     = float(np.corrcoef(signal, returns)[0, 1])
    partial_corr = float(np.corrcoef(x_res, y_res)[0, 1])

    retained = abs(partial_corr) / max(abs(raw_corr), 1e-8)
    passes = (p_val < alpha) and (retained >= 0.50)
```

---

## 3.4 · Step 3: DoWhy Refutation Tests

### 3.4.1 · Placebo Treatment Test

**Intuition:** "Replace the true signal with pure noise. If the noise signal achieves a comparable coefficient to the real signal, then the real signal has no genuine predictive content — it was just lucky. If the real signal is statistically distinct from noise at 5% significance, it passes."

$$p_{\text{placebo}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\!\left(|\hat{\theta}_b| \geq |\hat{\theta}_{\text{orig}}|\right), \quad \hat{X}^b_t \sim \mathcal{N}(0, \sigma^2_X)$$

**Pass condition:** $p_{\text{placebo}} > 0.05$ (original estimator is statistically distinct from noise).

### 3.4.2 · Policy Invariance Test (Structural Stability)

**Intuition:** "Train on the first half of the data. Test the structural relationship on the second half. Use Moving Block Bootstrap to get valid p-values under autocorrelation. If the relationship has shifted beyond what sampling variation can explain, the signal has a structural break."

$$p_{\text{policy}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\!\left(|\hat{\theta}^{(2)}_b - \hat{\theta}^{(1)}| \geq |\hat{\theta}^{(2)} - \hat{\theta}^{(1)}| + \hat{\Delta}\right)$$

where:

$$
\hat{\Delta} = \hat{\sigma}(\{\hat{\theta}^{(2)}_b\}_{b=1}^B)
$$

is the MBB standard deviation.

### 3.4.3 · Moving Block Bootstrap (MBB)

The standard bootstrap destroys autocorrelation. MBB resamples **blocks** of length $b=20$ to preserve the temporal dependence structure of macro time-series:

```
MOVING BLOCK BOOTSTRAP — WHY BLOCKS MATTER

  Standard bootstrap (invalid for time series):
  [r1, r5, r3, r8, r2, ...]  ← breaks autocorrelation structure

  MBB with block size b=20:
  [r1...r20, r45...r64, r12...r31, ...]  ← preserves local structure

  Why b=20? Captures typical macro autocorrelation length (~1 month)
  while being short enough to avoid resampling variance explosion.
```

**Code** — `causal.py`, `_moving_block_bootstrap()`:
```python
def _moving_block_bootstrap(data, block_size=20, n_reps=200, rng=None):
    n_blocks = math.ceil(t / block_size)
    samples = []
    for _ in range(n_reps):
        starts = rng.integers(0, max(max_start, 1), size=n_blocks)
        blocks = [data[s: s + block_size] for s in starts]
        resample = np.concatenate(blocks, axis=0)[:t]   # Trim to original T
        samples.append(resample)
    return samples
```

---

## 3.5 · Causal Confidence Factor γ

$$\gamma = \begin{cases} 0.95 & p_{\text{Granger}} < 0.05 \;\text{AND}\; \psi \geq 0.50 \;\text{AND}\; p_{\text{placebo}} > 0.05 \;\text{AND}\; p_{\text{policy}} > 0.05 \\ 0.30 & p_{\text{Granger}} < 0.05 \;\text{AND}\; (\psi < 0.50 \;\text{OR}\; p_{\text{policy}} \leq 0.05) \\ 0.00 & p_{\text{Granger}} \geq 0.05 \;\text{OR}\; p_{\text{placebo}} \leq 0.05 \end{cases}$$

**Final signal strength:** $\alpha^{\text{final}}_t = \hat{\alpha}_t \cdot \gamma$

```
γ DECISION TREE

  Step 1: Granger VARX
    p_Granger ≥ 0.05 → γ = 0.00 (REJECT)
    p_Granger < 0.05 → proceed to Step 2

  Step 2: CMI (alpha retained fraction ψ)
    ψ < 0.50 → γ = 0.30 (BETA_PROXY — reduced capital)
    ψ ≥ 0.50 → proceed to Step 3

  Step 3: DoWhy refutation
    p_placebo ≤ 0.05 → γ = 0.00 (REJECT — signal = noise)
    p_policy ≤ 0.05  → γ = 0.30 (BETA_PROXY — structural break)
    Both pass        → γ = 0.95 (PASS — full capital allocation)

  Capital allocation: w_i = γ * HRP_weight(rank_score_i)
```

**Code** — `causal.py`, `run_causal_stack()` and `rebalance.py`:
```python
# causal.py — γ routing
if not granger.passes:
    gamma = C.CAUSAL_GAMMA_REJECT    # 0.00
elif not cmi.passes:
    gamma = C.CAUSAL_GAMMA_MEDIUM    # 0.30
elif not dowhy.placebo_passes:
    gamma = C.CAUSAL_GAMMA_REJECT    # 0.00
else:
    gamma = C.CAUSAL_GAMMA_HIGH      # 0.95

# rebalance.py — γ gates the rebalancing decision
if causal_gamma < 0.25:
    go = False   # Suspend: causal validation failed
```

[🔝 Back to Top](#table-of-contents)

---
---

# §4 · Statistical Falsification Protocol

**Open with the intuition (15 seconds):**
> "The enemy of good quant research is the researcher degrees of freedom problem. If you test 100 parameters and pick the best, you will find a signal with SR = 3 that is pure noise. The five-step falsification protocol locks you in *before* you look at the data: pre-specified hypothesis, pre-specified test, pre-specified kill criteria. Anything else is data mining dressed up as research."

---

## 4.1 · Combinatorial Purged Cross-Validation (CPCV)

**Intuition:** "Standard k-fold cross-validation leaks information across folds because adjacent time periods are correlated. CPCV uses all possible combinations of non-contiguous test blocks, purges embargo regions at each boundary, and gives you 45 independent out-of-sample Sharpe paths from which you can build a robust OOS distribution."

Generate $\binom{N}{k}$ unique test paths:

$$\binom{10}{2} = 45 \text{ unique test paths}$$

Each path:
1. Selects $k=2$ non-contiguous blocks as test data
2. Purges $\delta = 1\%$ of each block from train data on both sides (embargo)
3. Computes SR on the test path only

```
CPCV vs WALK-FORWARD — WHY CPCV IS STRICTLY BETTER

  Walk-forward (wrong):               CPCV (correct):
  ─────────────────────────────────   ───────────────────────────────────
  Train:   [1....7]                   Train: [1,2,3,4,5,6,7,_,9,10]
  Test:    [8]                        Test:  [8]  (one of 45 paths)
                                      Embargo: [7] and [9] excluded
  1 test path.                        45 test paths.
  High variance OOS estimate.         Robust distribution of OOS SR.

  Why walk-forward fails:
  Data leaked from train period into test period via:
    - rolling features computed on adjacent time points
    - macro state variables that span the train/test boundary
```

The distribution $\{SR_1, \ldots, SR_{45}\}$ provides a robust OOS SR estimate free of in-sample bias.

**Code** — `falsification.py`, `combinatorial_purged_cv()`:
```python
def combinatorial_purged_cv(returns, signal, n_splits=10, n_test_splits=2, embargo_pct=0.01):
    combos = list(itertools.combinations(range(n_splits), n_test_splits))  # 45 paths
    for test_blocks in combos:
        # 1. Identify test indices + embargo zones
        # 2. Compute PnL: sig[test] * ret[test]
        # 3. SR = mean/std * sqrt(252)
        sr = mu / sigma * math.sqrt(C.ANNUALIZATION_FACTOR)
        sharpe_paths.append(sr)
    psr = float(np.mean(sr_arr >= C.SHARPE_FLOOR_SYSTEMATIC_MACRO))
```

---

## 4.2 · Multiple Testing Correction

**Intuition:** "We are testing M=2 signals. If we set α=5% for each, the family-wise error rate is 1 - (1-0.05)² ≈ 9.75% — nearly double. Bonferroni corrects for this at the cost of some power. Benjamini-Hochberg (BH) is more powerful when signals are correlated, controlling the false discovery rate rather than the family-wise error rate."

**Bonferroni (FWER):**

$$\alpha^{\*}_{\text{Bonf}} = \frac{0.05}{M} = \frac{0.05}{2} = 0.025$$

**Benjamini-Hochberg (FDR):**

Sort $p_{(1)} \leq \cdots \leq p_{(M)}$ and reject $H_{(i)}$ for:

$$
i \leq \max\lbrace j : p_{(j)} \leq \frac{j \alpha}{M}\rbrace
$$

BH controls FDR $\leq \alpha = 0.05$ under arbitrary dependence between test statistics.

```
BONFERRONI vs BH — WHEN EACH IS APPROPRIATE

  Bonferroni (conservative):          Benjamini-Hochberg (powerful):
  ─────────────────────────────────   ───────────────────────────────────
  Controls: FWER (P(any false         Controls: FDR (E[false discoveries /
  discovery) ≤ α)                     all discoveries])
  Use when: small M, high cost of     Use when: larger M, signal discovery
  any false positive (e.g., M=2)      matters more than strict control
  Properietary Trading Firm: M=2 → α* = 0.025              HLS: Same M=2, but BH reported
                                      alongside for audit trail
```

**Code** — `falsification.py`:
```python
def bonferroni_correction(pvalues):
    return np.asarray(pvalues) * len(pvalues)       # p_adj = p * M

def benjamini_hochberg(pvalues, alpha=0.05):
    order = np.argsort(raw)
    for i, idx in enumerate(order):
        if p[idx] <= alpha * (i + 1) / m:
            rejected[idx] = True
        else:
            break    # Step-up: once fails, all subsequent fail
```

---

## 4.3 · Deflated Sharpe Ratio

**Intuition:** "A SR of 2.0 sounds excellent. But if you ran 50 parameter combinations to find it, the SR you should benchmark against is not 0 — it is the expected maximum SR from random search across 50 trials. The DSR adjusts for this selection bias, for non-normality of returns (skewness and kurtosis reduce the true SR), and gives a probability that the strategy is genuinely above zero."

$$\text{DSR}(\widehat{SR}) = \Phi\!\left(\frac{(\widehat{SR} - SR^{\*})\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right)$$

where:
- $SR^{\*}$ = benchmark SR accounting for $M=2$ trial configurations
- $\hat{\gamma}_3$ = skewness of P&L (negative skew increases denominator, deflating DSR)
- $\hat{\gamma}_4$ = excess kurtosis (fat tails deflate DSR further)

**Pass condition:** DSR > 0.95 (95% probability the strategy is genuinely above zero SR).

```
SHARPE DECAY WATERFALL

  Gross backtest SR (pre-cost, full sample):  2.5
  ────────────────────────────────────────────────
  - Transaction costs haircut:               -0.40  (SHARPE_DECAY_TRANSACTION_COSTS)
  - Overfitting / IS-OOS gap:                -0.30  (SHARPE_DECAY_OVERFITTING)
  - Live slippage & signal decay:            -0.15  (SHARPE_DECAY_LIVE_SLIPPAGE)
  ────────────────────────────────────────────────
  Expected live net SR:                       1.65

  Month-6 KPI gate: walk-forward SR > 0.70
  (this is after all haircuts on a CPCV OOS distribution)
```

**Code** — `falsification.py`, `deflated_sharpe_ratio()`:
```python
def deflated_sharpe_ratio(sr, pnl, n_trials=1, alpha=0.05):
    skewness = float(_skew(pnl_arr))
    excess_kurt = float(_kurt(pnl_arr))
    # Mertens (2002) variance of SR estimator
    sr_var = (1.0 - skewness * sr + (excess_kurt - 1.0)/4.0 * sr**2) / (n - 1)
    z = (sr - sr_star) / math.sqrt(sr_var)
    return float(norm.cdf(z))
```

---

## 4.4 · Signal Half-Life via OU Regression

**Intuition:** "If the IC time series is mean-reverting (which it should be — no alpha lasts forever), we can model it as an Ornstein-Uhlenbeck process and estimate how long it takes for IC to decay by half. Below 21 days: the signal is decaying too fast to be worth the transaction costs. Above 63 days: the signal is too stale for 4× daily rebalancing."

$$dIC_t = \kappa(\mu - IC_t)\,dt + \sigma\,dW_t$$

Discretised via OLS: $\Delta IC_t = -\hat{\kappa}\, IC_{t-1} + \varepsilon_t$

$$T_{1/2} = \frac{\ln 2}{\hat{\kappa}}$$

```
SIGNAL RETIREMENT DECISION TABLE

  Condition               Threshold                   Action
  ─────────────────────   ─────────────────────────   ──────────────────
  Half-life (alert)       T½ < 17.85d (= 21 × 0.85)  RETIRE IMMEDIATELY
  Half-life (valid)       T½ ∈ [21, 63]d              Continue
  Half-life (stale)       T½ > 63d                    Review (too slow)
  IC floor                IC < 0.02                   RETIRE
  ICIR floor              ICIR < 0.50                 Reduce allocation
  Walk-forward Sharpe     SR < 0.70                   Reduce allocation
  Placebo p-value         p_placebo < 0.05            Suspend
```

**Code** — `falsification.py`, `estimate_half_life()`:
```python
def estimate_half_life(ic_series, min_obs=60):
    delta_ic = np.diff(ic)
    ic_lag = ic[:-1]
    result = np.linalg.lstsq(ic_lag.reshape(-1, 1), delta_ic, rcond=None)
    kappa = float(-result[0][0])
    half_life = math.log(2.0) / max(kappa, 1e-6)

    is_alive = C.ALPHA_HALF_LIFE_MIN <= half_life <= C.ALPHA_HALF_LIFE_MAX
    retirement_alert = half_life < C.ALPHA_HALF_LIFE_MIN * C.HALF_LIFE_BREACH_THRESHOLD
```

[🔝 Back to Top](#table-of-contents)

---
---

# §5 · Portfolio Construction

**Open with the intuition (15 seconds):**
> "Sample covariance matrices are noisy garbage in high dimensions. With T=1000 daily observations and N=50 assets, the sample covariance matrix has O(N²) parameters estimated from O(T×N) data — a fundamentally underdetermined problem. Ledoit-Wolf regularises this analytically. HRP then builds on this clean covariance matrix to construct weights that are robust to estimation error without requiring explicit mean estimates."

---

## 5.1 · Ledoit-Wolf Covariance Shrinkage

**Intuition:** "The Ledoit-Wolf estimator is the solution to a supervised learning problem: find the linear combination of the sample covariance matrix and a structured target that minimises the expected Frobenius loss from the true covariance. It has a closed-form solution — no cross-validation required."

$$
\hat{\Sigma}^{\text{LW}} = (1 - \hat{\alpha}^{\*})\hat{\Sigma}^{\text{sample}} + \hat{\alpha}^{\*} F
$$

where $F = \frac{\text{tr}(\hat{\Sigma})}{N} I$ is the identity shrinkage target, and $\hat{\alpha}^{\*}$ is the analytically optimal shrinkage intensity under Frobenius loss $\|\hat{\Sigma} - \Sigma\|^2_F$.

```
WHY SHRINKAGE IS NECESSARY

  T=252, N=50 (one year daily, 50 assets):

  Sample covariance (wrong):          Ledoit-Wolf (correct):
  ─────────────────────────────────   ───────────────────────────────────
  Extreme eigenvalues: λ_max = 8.3    Extreme eigenvalues pulled toward
  λ_min = 0.02 (near singular!)       mean: λ_max = 4.1, λ_min = 0.15
  Inverse is numerically unstable     Stable, well-conditioned inverse
  Portfolio weights blow up           Smooth, diversified weights
  Sharpe in backtest: 3.2             Sharpe in backtest: 1.8 (honest)
```

**Code** — `portfolio.py`, `ledoit_wolf_shrink()`:
```python
def ledoit_wolf_shrink(returns_matrix):
    try:
        from sklearn.covariance import OAS   # Oracle Approximating Shrinkage
        oas = OAS()
        oas.fit(returns_matrix)
        return oas.covariance_              # Analytically optimal
    except ImportError:
        # Manual LW fallback: α* = (tr(S²) + tr(S)²) / ((T+1-2/N)(tr(S²) - tr(S)²/N))
        mu = trace_s / n
        target = mu * np.eye(n)
        return delta * target + (1.0 - delta) * s
```

---

## 5.2 · Hierarchical Risk Parity

**Intuition:** "Mean-variance optimisation requires a precise estimate of expected returns — which we do not have. HRP sidesteps this by using only the covariance matrix. It clusters assets by correlation distance, builds a hierarchy of increasingly diversified portfolios via Ward linkage, and allocates weights via recursive bisection. The result is a portfolio that is robust to covariance estimation error and avoids the instability of MV optimisation."

The HRP algorithm:
1. Compute Ledoit-Wolf covariance $\hat{\Sigma}^{\text{LW}}$
2. Compute correlation distance: $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$
3. Ward hierarchical clustering on the condensed distance matrix
4. Recursive bisection: allocate proportional to inverse-variance within each cluster branch
5. Apply signal-score tilt (long-short construction)
6. Cap single positions at $w_{\max} = 5\%$

```
HRP DENDROGRAM — ILLUSTRATIVE (8 commodity assets)

         ┌──────────────────────────────────────┐
         │                                      │
    ┌────┘                                ┌─────┘
    │                                     │
  ┌─┘     ┌────┐                    ┌───┐  └────┐
  │       │    │                    │   │       │
  WTI   BRENT NGAS              COPPER GOLD  SILVER

  Energy cluster          Metals cluster
  (high intra-correlation) (moderate correlation)

  HRP allocates proportionally to inverse-variance:
  Energy cluster: ~35% (higher vol assets get less)
  Metals cluster: ~65%
  Within cluster: inverse-variance weighting
```

**Code** — `portfolio.py`, `hierarchical_risk_parity()`:
```python
def hierarchical_risk_parity(returns_matrix, signal_scores):
    cov = ledoit_wolf_shrink(returns_matrix)
    dist = np.sqrt(0.5 * (1.0 - corr))                     # Correlation distance
    linkage = hierarchy.linkage(dist_condensed, method="ward")  # Ward clustering
    # Recursive bisection
    while cluster_list:
        left, right = split(current)
        alpha = 1.0 - var_left / (var_left + var_right)    # Bisection weight
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
    # Signal tilt: positive score → long, negative → short
    hrp_weights = weights * signal_norm
```

---

## 5.3 · Bayesian Signal Integration — Black-Litterman

**Intuition:** "Once both signals are in production (Month 4), we combine them via Black-Litterman. The key innovation: view confidence is not a subjective guess — it is the product of the Deflated Sharpe Ratio and the causal confidence factor γ. This creates a theoretically principled link between our validation framework and portfolio sizing."

$$
\mu^{\text{BL}} = \left[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau\Sigma)^{-1} \Pi + P^T \Omega^{-1} q\right]
$$

where:

$$
\Omega^{-1} = \text{diag}(\text{DSR}_{\text{ISCF}}, \ \text{DSR}_{\text{MGD}}) \cdot \gamma
$$

scales view confidence by both statistical validation and causal confidence.

```
BLACK-LITTERMAN VIEW CONFIDENCE MATRIX Ω^{-1}

  Signal    DSR    γ (causal)   Ω^{-1} entry   Interpretation
  ────────  ─────  ───────────  ─────────────  ──────────────────────────────
  ISCF      0.97   0.95         0.92           High confidence — full weight
  MGD       0.89   0.30         0.27           Moderate — beta-proxy only
  MGD       0.89   0.95         0.85           High confidence (after γ pass)
  Either    any    0.00         0.00           Signal suspended — no BL view
```

[🔝 Back to Top](#table-of-contents)

---
---

# §6 · 4× Daily Rebalancing Engine

**Open with the intuition (15 seconds):**
> "Four sessions per day: Asia, London, NY Open, NY Close. At each session, the engine computes target weights from current signal scores, applies γ scaling, estimates transaction costs via Almgren-Chriss square-root impact, and fires a Go/No-Go decision. The three kill criteria — γ < 0.25, turnover > 50%, TC > 50% of expected alpha — prevent the engine from trading when costs exceed benefits."

Target weight at each session $s$:

$$w^{\text{target}}_{i,s} = \gamma \cdot \frac{|r_{i,s}|}{\sum_j |r_{j,s}|}$$

Transaction cost model (Almgren-Chriss square-root impact):

$$\text{TC (bps)} = \frac{\text{spread}}{2} \cdot \Delta w \cdot 10^4 + \lambda_{\text{impact}} \sqrt{\frac{\Delta w}{\text{ADV fraction}}}$$

Go/No-Go condition: rebalance if and only if:

$$\gamma \geq 0.25 \quad\text{AND}\quad \text{turnover} \leq 0.50 \quad\text{AND}\quad \text{TC} \leq 0.50 \times \alpha_{\text{expected}}$$

```
SESSION SCHEDULE — 4× DAILY REBALANCING

  Session       UTC        Primary Markets       Signal Emphasis
  ────────────  ─────────  ────────────────────  ──────────────────────────────
  Asia          00:00 UTC  JPY, AUD, metals      ISCF: base metals inventory
                           (LME 09:00 HKT)       MGD: BoJ, RBA statements
  London        07:00 UTC  EUR, GBP, bunds       MGD: flash PMI releases
                           (LME ring open)       ISCF: energy basis UK
  NY Open       12:00 UTC  USD, CAD, crude oil   MGD: NFP, CPI releases
                           (CME open)            ISCF: WTI/Brent basis
  NY Close      21:00 UTC  All G10 FX            Health: IC/SR monitoring
                           (end of day)          Rebalance: settlement prices
```

**CI/CD cron schedule** (`.github/workflows/ci.yml`):
```yaml
schedule:
  - cron: '0 7 * * 1-5'    # London open
  - cron: '0 12 * * 1-5'   # NY open
  - cron: '0 17 * * 1-5'   # London close
  - cron: '0 21 * * 1-5'   # NY close / Asia open
```

**Code** — `rebalance.py`, `compute_rebalance()`:
```python
def compute_rebalance(signal_name, rank_scores, current_weights, causal_gamma, session, ...):
    target = causal_gamma * raw_target       # γ-scaled weights

    # TCA: bid-ask + sqrt-law impact
    impact_bps = 10.0 * math.sqrt(max(turnover / adv_fraction, 0))
    tc_bps = bid_ask_bps / 2.0 * turnover * 1e4 + impact_bps

    # Kill criteria (pre-agreed — cannot be overridden)
    if causal_gamma < 0.25:
        go = False   # Causal validation suspended
    elif turnover > 0.50:
        go = False   # Split across 2 sessions
    elif tc_bps > 0.5 * expected_alpha_bps:
        go = False   # TC > 50% of alpha: not worth it
    else:
        go = True
```

[🔝 Back to Top](#table-of-contents)

---
---

# §7 · Implementation Architecture

**A guided tour of every production file, with the dissertation formula it implements.**

---

### Architecture Overview

```
hls_trading_six_month_plan/
├── src/
│   ├── python/citadel_alpha/
│   │   ├── constants.py        ← All empirical constants (§0-§8 calibration)
│   │   ├── signals.py          ← Base classes: SignalResult, rank norm, ICIR
│   │   ├── signals_hls.py      ← ISCF + MGD implementation (§1-§2)
│   │   ├── causal.py           ← 3-step causal stack (§3)
│   │   ├── falsification.py    ← CPCV, DSR, BH, half-life (§4)
│   │   ├── portfolio.py        ← HRP + Ledoit-Wolf + BL (§5)
│   │   ├── rebalance.py        ← 4× session engine + TCA (§6)
│   │   ├── data_provider.py    ← Design-by-Contract provider ABCs
│   │   └── analytics.py        ← FLOAM, orthogonality, HMM regimes
│   └── cpp/
│       ├── alpha_engine.hpp    ← C++26 hot-path signal headers
│       ├── alpha_engine.cpp    ← MAD, GaussianRankNorm, SpearmanCorr
│       └── bindings.cpp        ← nanobind Python ↔ C++ bindings
├── .github/workflows/ci.yml   ← 4× daily CI cron, KPI monitoring
├── tests/python/               ← Pytest suite (signals, causal, portfolio)
└── pyproject.toml              ← Build config (Python 3.13, uv)
```

---

### `constants.py` — The Single Source of Truth

Every empirical constant used across the codebase lives here. The dissertation's parameters map directly:

```python
# constants.py — Properietary Trading Firm Branch Signal Constants
ISCF_MAX_BASIS_ZSCORE: float = 4.0          # z_max winsorisation cap (§2.2.2)
ISCF_VOL_NORMALISATION_WINDOW: int = 20     # H = 20 in σ^rv (§2.2.1)
MGD_PMI_WEIGHT: float = 0.40                # w^PMI (§3.2.1)
MGD_INFLATION_WEIGHT: float = 0.30          # w^CPI (§3.2.1)
MGD_EMPLOYMENT_WEIGHT: float = 0.30         # w^EMP (§3.2.1)
MGD_SURPRISE_EMA_SPAN: int = 5              # τ_EMA (§3.2.2)
CAUSAL_GAMMA_HIGH: float = 0.95             # γ full pass (§4.4.4)
CAUSAL_GAMMA_MEDIUM: float = 0.30           # γ beta-proxy (§4.4.4)
CAUSAL_GAMMA_REJECT: float = 0.0            # γ reject (§4.4.4)
NEWEY_WEST_LAG_TRUNCATION: int = 12         # L HAC bandwidth (§4.2)
BLOCK_BOOTSTRAP_BLOCK_SIZE: int = 20        # b MBB block size (§4.4.3)
ALPHA_HALF_LIFE_MIN: int = 21               # T½ floor (§6)
ALPHA_HALF_LIFE_MAX: int = 63               # T½ ceiling (§6)
IC_FLOOR: float = 0.02                      # IC kill threshold (§5.2)
```

---

### `signals_hls.py` — ISCF and MGD Signal Construction

This is the core signal module. The full ISCF pipeline (basis → MAD z-score → idiosyncratic extraction → Gram-Schmidt → rank normalisation) is expressed in roughly 50 lines of NumPy:

```python
def compute_iscf(spot, deferred, rvol, next_ret, macro_beta, baseline_factors):
    # Step 1: Volatility-normalised basis z-score (§2.2.1-2)
    basis_z = _compute_basis_zscore(spot, deferred, rvol)
    beta = np.clip(macro_beta, 0.0, 1.0)

    # Step 2: Idiosyncratic extraction (§2.2.3)
    # α = sign(z) * sqrt(|z|) * (1 - β_macro)
    raw = np.sign(basis_z) * np.sqrt(np.abs(basis_z)) * (1.0 - beta)

    # Step 3: Gram-Schmidt orthogonalisation (§2.2.4)
    raw_orth = gram_schmidt_residualise(raw, baseline_factors)

    # Step 4: Cross-sectional z-score + Gaussian rank normalisation (§2.2.5)
    z = (raw_orth - mu) / max(sigma, 1e-8)
    rank = _gaussian_rank_normalise(z)
    ...
```

---

### `causal.py` — Three-Step Causal Stack

The `run_causal_stack()` function orchestrates the full pipeline in one call, returning a `CausalStackResult` with a human-readable summary printed to the monitoring dashboard:

```
Signal: ISCF
  Step 1 — Granger VARX : F=4.821 p=0.0183 lag=2 → ✓ PASS
  Step 2 — CMI          : stat=0.1843 retained=71.20% → ✓ PASS
  Step 3 — DoWhy Placebo: p=0.1750 → ✓ PASS
  Step 3 — Policy Inv.  : p=0.2240 → ✓ PASS
  γ (Causal Confidence) : 0.95
  Recommendation        : ✓ PASS
```

---

### `falsification.py` — Pre-Registered Falsification

The `signal_health_report()` function produces the retirement decision at every monitoring cycle:

```
Signal: ISCF
  Mean IC       : 0.0312  (✓ floor=0.02)
  ICIR          : 0.7184  (✓ floor=0.50)
  Half-life     : 41.3d   (✓ range=[21,63]d)
  Gross SR      : 2.241   (✓ floor=2.0)
  t-stat        : 5.12    (✓ floor=3.0)
  Net SR (est.) : 1.341
  RETIRE?       : ✓ NO
```

---

### `data_provider.py` — Design-by-Contract Plug-In Architecture

The `AbstractISCFProvider` and `AbstractMGDProvider` ABCs enforce a contract with pre- and postconditions:

```python
class AbstractISCFProvider(abc.ABC):
    @abc.abstractmethod
    def fetch(self, start, end, assets=None) -> ISCFMarketData:
        """Preconditions: start < end
           Postconditions: no NaN, spot > 0, rvol > 0, macro_beta ∈ [0,1]"""
        ...

    def validate(self, data: ISCFMarketData) -> None:
        assert np.all(data.spot > 0), "spot must be positive"
        assert np.all(data.rvol > 0), "rvol must be positive"
        assert np.all((data.macro_beta >= 0) & (data.macro_beta <= 1))
```

This guarantees that `YFinanceISCFProvider` and `HLSISCFProvider` are **drop-in replacements** — the signal construction code never needs to change when HLS infrastructure is available.

---

### C++26 Hot-Path Engine — `alpha_engine.cpp`

The Python implementation is the reference. The C++ engine provides the same computation for latency-sensitive production paths, exposed via `nanobind`. The `MedianAbsDev()` function is the C++ equivalent of the Python MAD estimator:

```cpp
double MedianAbsDev(std::span<const double> x) {
    // Sort a copy, find median, compute |x - median|, sort again, find MAD
    std::sort(sorted.begin(), sorted.end());
    double median = (n % 2 == 0)
        ? 0.5 * (sorted[n/2-1] + sorted[n/2])
        : sorted[n/2];
    for (size_t i = 0; i < n; ++i) abs_dev[i] = std::abs(sorted[i] - median);
    std::sort(abs_dev.begin(), abs_dev.end());
    return (n % 2 == 0) ? 0.5*(abs_dev[n/2-1]+abs_dev[n/2]) : abs_dev[n/2];
}
```

The Python layer checks for the C++ extension at import time with a graceful fallback:
```python
# signals.py
try:
    from citadel_alpha import _citadel_alpha_cpp as _cpp  # type: ignore[import]
    _CPP_AVAILABLE = True
    logger.info("C++ alpha engine loaded.")
except ImportError:
    _cpp = None
    _CPP_AVAILABLE = False
    logger.warning("C++ extension not found; using pure-Python fallback.")
```

[🔝 Back to Top](#table-of-contents)

---
---

# §8 · Six-Month Deployment Plan

```
MONTH   MILESTONE                TASKS                                           KPI GATE
──────  ───────────────────────  ──────────────────────────────────────────────  ────────────────────────
1       Framework Audit          Map signal infrastructure; ingest LME/CME       Sharpe decay
        + Data Ingestion         prompt-date spreads + freight (ISCF);           baseline computed
                                 flash PMI, inflation, CB feeds (MGD);
                                 compute live vs backtest SR decay curves.
                                 [data_provider.py → YFinance/FRED free tier]

2       Signal Construction      Build ISCF basis/vol factor; build MGD          |R²| < 0.15
        + Orthogonalisation      composite surprise; rolling Gram-Schmidt         (analytics.py
                                 against [trend, mom, carry]; verify R² < 0.15;  orthogonality check)
                                 IC uncorrelated with existing 5 factors.
                                 [signals_hls.py complete]

3       Statistical Validation   CPCV (45 paths, N=10, k=2); DSR (M=2);         DSR > 0;
                                 BH/Bonferroni; DoWhy placebo + MBB policy        all 3 causal
                                 invariance; causal γ assignment;                 steps pass
                                 pre-register kill criteria.
                                 [causal.py + falsification.py complete]

4       Portfolio Ensemble       BL integration with DSR/γ priors; real-time     Paper SR > 1.0
                                 correlation monitoring (regime-shift alerts at
                                 |Δρ| > 0.20); TCA build for Metals/Energy/FX
                                 microstructures; 4× daily rebalancing engine.
                                 [portfolio.py + rebalance.py complete]

5       Paper-Trading            Shadow production on live vendor feeds;          Fill rate > 95%
        + Grading                execution audit (fill rate, slippage vs TCA);
                                 CI/CD cron health dashboard running.
                                 [.github/workflows/ci.yml 4× daily cron active]

6       Production Launch        Live capital allocation; health dashboard        Walk-forward SR
                                 (SR decay, T½, regime, γ); final FDR            > 0.70;
                                 verification.                                    FDR ↓ ≥ 30%
                                 [generate_kpi_report.py → GITHUB_STEP_SUMMARY]
```

### Month-6 KPIs (from `constants.py`)

```python
WALKFORWARD_SHARPE_TARGET: float = 0.70    # Walk-forward SR > 0.70
CORRELATION_SUPPRESSION_TARGET: float = 0.15  # max |ρ| with legacy < 0.15
FDR_REDUCTION_TARGET: float = 0.30        # Portfolio FDR ↓ ≥ 30%
```

### CI/CD Signal Health Monitor

The GitHub Actions workflow runs the full health check at every rebalancing session and posts a KPI report to the job summary:

```yaml
# .github/workflows/ci.yml — signal-health-monitor job
- name: Run ISCF + MGD signal health check
  run: |
    hls-alpha hls-monitor \
      --n-assets 8 \
      --n-periods 2000 \
      --output-dir artifacts \
      --sr-floor 0.70

    python3 scripts/generate_kpi_report.py \
      --artifacts-dir artifacts \
      --output artifacts/KPI_REPORT.md \
      --session "$(date -u +'%Y-%m-%d %H:%M UTC')"
```

Slack alerts fire on test failure or retirement trigger, ensuring the research team is notified within minutes of any signal degradation.

[🔝 Back to Top](#table-of-contents)

---
---

# §9 · Quick-Reference Equation Sheet

### Signal 1 — ISCF

| Step | Formula | Code Reference |
|---|---|---|
| No-arbitrage futures | $`F(t,T) = S(t)\,e^{(r+u-c)(T-t)}`$ | Theory of storage (Working 1949) |
| Convenience yield | $`c = g_0 + g_1/I_t \quad \text{when } I_t \leq I^{\text{*}}`$ | `constants.py: ISCF_INVENTORY_DECAY_LAMBDA` |
| Vol-normalised basis | $`b_{i,t} = (S_{i,t} - F^{\text{def}}_{i,t})\,/\,\max(\sigma^{rv}_{i,t}, \varepsilon)`$ | `signals_hls.py: _compute_basis_zscore()` |
| 20d realised vol | $`\sigma^{rv}_{i,t} = \sqrt{252/H \sum_{h=1}^{H} (\log S_{i,t-h+1}/S_{i,t-h})^2}`$ | `signals_hls.py: _compute_basis_zscore()` |
| MAD robust z-score | $`z_{i,t} = \text{clip}((b_{i,t} - \text{med}_j b_j)\,/\,(\text{MAD}_j + \varepsilon),\,-4, +4)`$ | `signals_hls.py: _compute_basis_zscore()` |
| Idiosyncratic extraction | $`\alpha^{\text{raw}}_{i,t} = \text{sign}(z_{i,t})\,\\|z_{i,t}\\|^{1/2}\,(1-\beta^{\text{macro}}_{i,t})`$ | `signals_hls.py: compute_iscf()` |
| Gram-Schmidt | $`\hat{\alpha}^{\text{ISCF}}_t = \alpha^{\text{raw}}_t - \sum_k \frac{\langle \alpha^{\text{raw}}, f^k\rangle}{\langle f^k, f^k\rangle} f^k`$ | `signals_hls.py: gram_schmidt_residualise()` |
| Gaussian rank norm | $`r_{i,t} = \Phi^{-1}(\text{rank}(\hat{\alpha}_{i,t})/(N+1))`$ | `signals.py: _gaussian_rank_normalize()` |

### Signal 2 — MGD

| Step | Formula | Code Reference |
|---|---|---|
| Rational expectations | $`\mathbb{E}_t[S_{t+h}] = F_{t,h}`$ | Muth (1961) |
| Composite surprise | $`\mathcal{S}_{i,t} = 0.40\,\Delta\text{PMI} + 0.30\,\Delta\text{CPI} + 0.30\,\Delta\text{EMP}`$ | `signals_hls.py: _composite_surprise()` |
| Standardised surprise | $`\Delta^k_{i,t} = (\text{actual} - \text{consensus})\,/\,\sigma^k_{\text{hist}}`$ | `data_provider.py: YFinanceMGDProvider` |
| EMA (Kalman steady-state) | $`\hat{\mu}_t = \alpha \mathcal{S}_t + (1-\alpha)\hat{\mu}_{t-1},\; \alpha = 2/(\tau+1)`$ | `signals_hls.py: compute_mgd()` |
| Divergence signal | $`D_{i,t} = (\mathcal{S}_{i,t} - \hat{\mu}_{i,t})\,/\,\max(\sigma^{60}_{i,t}, \varepsilon)`$ | `signals_hls.py: compute_mgd()` |

### Causal Validation Framework

| Step | Test | Pass Condition | Code Reference |
|---|---|---|---|
| 1: VARX Granger | $`F = (RSS_R - RSS_U)/df_1 \;/\; \hat{\sigma}^2_{\text{HAC}}`$ | $`p < 0.05`$ | `causal.py: granger_causality_varx()` |
| 1: Newey-West HAC | $`\hat{\Sigma}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_j (\frac{1-j}{(L+1)})(\hat{\Gamma}_j + \hat{\Gamma}_j^T)`$ | $`L = \lfloor 4(T/100)^{2/9}\rfloor`$ | `causal.py: _newey_west_variance()` |
| 2: CMI proxy | $`\psi = \frac{\\|\rho_{\text{partial}}\\|}{\\|\rho_{\text{raw}}\\|}`$ | $`\psi \geq 0.50`$ | `causal.py: conditional_independence_test()` |
| 3a: Placebo | $`p_{\text{pl}} = B^{-1}\sum_b \mathbf{1}(\\|\hat{\theta}_b\\| \geq \\|\hat{\theta}_{\text{orig}}\\|)`$ | $`p_{\text{pl}} > 0.05`$ | `causal.py: dowhy_refutation()` |
| 3b: Policy inv. | MBB structural stability test on regime split | $`p_{\text{pol}} > 0.05`$ | `causal.py: dowhy_refutation()` |
| γ assignment | $`\gamma \in \{0.00, 0.30, 0.95\}`$ | All pass → 0.95 | `causal.py: run_causal_stack()` |

### Statistical Validation

| Test | Formula | Code Reference |
|---|---|---|
| CPCV paths | $`\binom{10}{2} = 45`$ test paths | `falsification.py: combinatorial_purged_cv()` |
| Bonferroni | $`\alpha^{\text{*}} = \frac{0.05}{M}`$ | `falsification.py: bonferroni_correction()` |
| Deflated SR | $`\text{DSR} = \Phi\!\left((\widehat{SR}-SR^{\text{*}})\sqrt{T-1}\,/\,\sqrt{1-\hat{\gamma}_3\widehat{SR}+(\hat{\gamma}_4-1)/4\cdot\widehat{SR}^2}\right)`$ | `falsification.py: deflated_sharpe_ratio()` |
| Half-life | $`T_{1/2} = \ln 2\,/\,\hat{\kappa} \quad \text{from } \Delta IC_t = -\hat{\kappa}\,IC_{t-1}`$ | `falsification.py: estimate_half_life()` |

### Portfolio Construction

| Component | Formula | Code Reference |
|---|---|---|
| Fundamental Law | $`\text{IR} = \text{IC}\cdot\sqrt{K\cdot\text{Breadth}_{\text{single}}}`$ | `analytics.py: compute_floam()` |
| Ledoit-Wolf | $`\hat{\Sigma}^{\text{LW}} = (1-\hat{\alpha}^{\text{*}})\hat{\Sigma}^{\text{sample}} + \hat{\alpha}^{\text{*}} (\text{tr}(\hat{\Sigma})/N)\,I`$ | `portfolio.py: ledoit_wolf_shrink()` |
| HRP distance | $`d_{ij} = \sqrt{0.5(1-\rho_{ij})}`$ | `portfolio.py: hierarchical_risk_parity()` |
| Black-Litterman | $`\mu^{\text{BL}} = [(\tau\Sigma)^{-1}+P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi+P^T\Omega^{-1}q]`$ | `portfolio.py: deflated_sharpe_ratio()` |
| BL view confidence | $`\Omega^{-1} = \text{diag}(\text{DSR}_{\text{ISCF}}, \text{DSR}_{\text{MGD}})\cdot\gamma`$ | Dissertation §7.2 |
| TCA model | $`\text{TC(bps)} = \frac{\text{spread}}{2}\Delta w \cdot 10^4 + \lambda\sqrt{\Delta w\,/\,\text{ADV}}`$ | `rebalance.py: compute_rebalance()` |
| Rebalancing gate | $`\gamma \geq 0.25 \;\text{AND}\; \text{TO} \leq 0.50 \;\text{AND}\; \text{TC} \leq 0.50\,\alpha_{\text{exp}}`$ | `rebalance.py: compute_rebalance()` |

---

### Performance Targets Summary

```
METRIC                     TARGET              SOURCE (constants.py)
─────────────────────────  ──────────────────  ──────────────────────────────────
Pre-cost SR                ≥ 2.0               SHARPE_FLOOR_SYSTEMATIC_MACRO
Walk-forward SR (Month 6)  > 0.70              WALKFORWARD_SHARPE_TARGET
Signal half-life           T½ ∈ [21, 63]d      ALPHA_HALF_LIFE_MIN/MAX
IC floor                   ≥ 0.02              IC_FLOOR
ICIR floor                 ≥ 0.50              ICIR_FLOOR
t-statistic                ≥ 3.0               TSTAT_SIGNIFICANCE
Max baseline R²            < 0.15              MAX_R2_ORTHOGONALITY
Portfolio FDR reduction    ≥ 30%               FDR_REDUCTION_TARGET
Max single-position weight  ≤ 5%               POSITION_LIMIT_MAX
Max ADV participation      ≤ 10%               ADV_PARTICIPATION_CAP
```

---

[🔝 Back to Top](#table-of-contents)

---
---

*Quantitative Research Pod · 2025*
*ISCF & MGD: Systematic Macro Alpha from First Principles*

---
---

[↩️ Back to ../README.md](../README.md)

---
---
