# 🤖 AI + ML NEXT FRONTIER — Automated Alpha Research Pipeline
### ISCF & MGD as Test Bench · Generic Framework · Human Oversight Architecture
#### Shaikat Majumdar · HLS Trading Quantitative Researcher Interview

> **Purpose:** Demonstrate a concrete, production-minded roadmap for compressing
> the ideation-to-P&L cycle using AI + ML — with ISCF and MGD as the test bench,
> full generalisation to arbitrary signal types, and principled human-in-the-loop
> safeguards against hallucination and overfitting.

---

## 📑 Table of Contents

- [§0 · The Problem: Ideation-to-P&L Latency](#0--the-problem-ideation-to-pl-latency)
- [§1 · Framework Overview: The AI Alpha Factory](#1--framework-overview-the-ai-alpha-factory)
- [§2 · Module A — AI-Assisted Signal Ideation Engine](#2--module-a--ai-assisted-signal-ideation-engine)
- [§3 · Module B — Automated Feature Engineering & Signal Construction](#3--module-b--automated-feature-engineering--signal-construction)
- [§4 · Module C — ML Validation & Anti-Overfitting Stack](#4--module-c--ml-validation--anti-overfitting-stack)
- [§5 · Module D — Multi-Agent Research Orchestration](#5--module-d--multi-agent-research-orchestration)
- [§6 · Module E — Human Oversight & Hallucination Guardrails](#6--module-e--human-oversight--hallucination-guardrails)
- [§7 · ISCF & MGD as Test Bench](#7--iscf--mgd-as-test-bench)
- [§8 · Generalisation: Making the Framework Signal-Agnostic](#8--generalisation-making-the-framework-signal-agnostic)
- [§9 · Implementation Roadmap: Months 1–6](#9--implementation-roadmap-months-16)
- [§10 · Key Equations & Complexity Analysis](#10--key-equations--complexity-analysis)

---

## §0 · The Problem: Ideation-to-P&L Latency

[🔝 Back to Top](#-table-of-contents)

### 🗣️ Feynman Version
> "A quant researcher has a great idea on Monday. After months of data gathering,
> coding, backtesting, and reviewing, the strategy goes live the following year —
> if it survives. Most of the calendar time is **waiting**: waiting for data,
> waiting for code, waiting for someone to review the stats. AI and ML can
> eliminate most of that wait, compressing a 6-month cycle into 6 weeks."

### The Economics of Speed

In a competitive alpha space, signal half-life decays with crowding:

$$\frac{d\,IC(t)}{dt} = -\kappa \cdot IC(t) - \delta \cdot C(t) \cdot IC(t)$$

where $\kappa$ is natural decay, $C(t)$ is crowding pressure (# of strategies
exploiting same anomaly), and $\delta$ is decay-per-crowding-unit.

*Non-mathematically: the faster you move from idea to production, the more of the
half-life you capture before competitors crowd you out. Slow pipelines destroy alpha.*

```
IDEATION-TO-P&L LATENCY: CURRENT vs. AI-ASSISTED

 Current (manual):
  Week  1-2:  Hypothesis formulation
  Week  3-4:  Data sourcing & cleaning
  Week  5-10: Signal construction + backtest
  Week 11-14: Statistical validation
  Week 15-20: Code review, paper doc, PM approval
  Week 21-26: Production deployment
  ─────────────────────────────────────────────
  Total:       ~26 weeks  (6 months)

 AI-Assisted target:
  Day   1-3:  AI ideation + hypothesis scoring
  Day   4-7:  Automated feature engineering
  Day   8-14: Parallel backtesting + validation
  Day  15-21: Multi-agent review + human gate
  Day  22-35: Staged deployment
  ─────────────────────────────────────────────
  Total:       ~5 weeks   (87% cycle reduction)
```

---

## §1 · Framework Overview: The AI Alpha Factory

[🔝 Back to Top](#-table-of-contents)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI ALPHA FACTORY — SYSTEM OVERVIEW                   │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──── MODULE A ─────────────────────────────────────────────────────────┐
  │  IDEATION ENGINE                                                       │
  │  LLM (fine-tuned on quant finance corpus)                              │
  │  + Knowledge Graph (factor taxonomy, market microstructure)            │
  │  + KPI Scoring (IC target, IR hurdle, orthogonality constraint)        │
  │  → Ranked hypothesis list with economic narrative + falsification plan │
  └──────────────────────────┬────────────────────────────────────────────┘
                             │ Approved hypotheses (human gate #1)
                             ▼
  ┌──── MODULE B ─────────────────────────────────────────────────────────┐
  │  FEATURE ENGINEERING AGENT                                             │
  │  AutoML feature synthesis (tsfresh, featuretools, custom transforms)   │
  │  + Domain-constrained search (no look-ahead, no cross-contamination)   │
  │  + Automatic orthogonality check vs. existing factor library           │
  │  → Feature store with IC pre-screening                                 │
  └──────────────────────────┬────────────────────────────────────────────┘
                             │ Feature sets (VIF < 5 gate)
                             ▼
  ┌──── MODULE C ─────────────────────────────────────────────────────────┐
  │  ML VALIDATION STACK                                                   │
  │  CPCV (45 paths) + DSR + BH correction                                 │
  │  + Regime stress test (2008, 2020, 2022 replay)                        │
  │  + Walk-forward OOS + Monte Carlo permutation                          │
  │  → Signal scorecard: IC, ICIR, DSR, half-life, regime sensitivity      │
  └──────────────────────────┬────────────────────────────────────────────┘
                             │ Scorecard (human gate #2)
                             ▼
  ┌──── MODULE D ─────────────────────────────────────────────────────────┐
  │  MULTI-AGENT ORCHESTRATOR                                              │
  │  Critic Agent    → attacks the hypothesis (adversarial)                │
  │  Stats Agent     → reruns validation independently                     │
  │  Risk Agent      → tail-scenario sizing and drawdown analysis          │
  │  Narrative Agent → drafts human-readable research memo                 │
  │  Coordinator     → resolves conflicts, escalates to human              │
  └──────────────────────────┬────────────────────────────────────────────┘
                             │ Consensus report (human gate #3 — PM)
                             ▼
  ┌──── MODULE E ─────────────────────────────────────────────────────────┐
  │  HUMAN OVERSIGHT & GUARDRAILS                                          │
  │  Hallucination detector (fact-checks all LLM claims against data)      │
  │  Statistical sanity checker (flags "too good to be true" results)      │
  │  Audit trail (immutable log of every AI decision + data source)        │
  │  → Go/No-Go for production; or → loop back to Module A                 │
  └──────────────────────────┬────────────────────────────────────────────┘
                             │ Production-approved signal
                             ▼
  ┌──── PRODUCTION ───────────────────────────────────────────────────────┐
  │  C++26 hot-path signal engine (existing HLS infrastructure)            │
  │  4× daily rebalancing with γ-gated position sizing                     │
  │  Continuous IC monitoring → automated retirement trigger               │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## §2 · Module A — AI-Assisted Signal Ideation Engine

[🔝 Back to Top](#-table-of-contents)

### Architecture

The ideation engine is a **retrieval-augmented generation (RAG)** system grounded
in a curated quant finance knowledge base, constrained to output *falsifiable*,
*economically-grounded* hypotheses — not pattern-mined noise.

```
IDEATION ENGINE INTERNALS

  ┌─────────────────────────────────────────────────────────┐
  │  KNOWLEDGE BASE (vector store)                          │
  │  ├─ Academic corpus: 500+ quant finance papers          │
  │  ├─ Factor taxonomy: carry, momentum, value, quality... │
  │  ├─ Existing signal library (ISCF, MGD + 28 others)     │
  │  ├─ Market microstructure encyclopedia                  │
  │  └─ Regime database: 2000, 2008, 2020, 2022 episodes    │
  └──────────────────┬──────────────────────────────────────┘
                     │ Retrieval (FAISS k-NN, k=20)
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │  LLM (fine-tuned on quant corpus)                       │
  │  System prompt enforces:                                │
  │  1. Economic mechanism FIRST (not statistical pattern)  │
  │  2. Explicit falsification criterion                    │
  │  3. Data requirements declared upfront                  │
  │  4. Orthogonality argument vs. existing factors         │
  └──────────────────┬──────────────────────────────────────┘
                     │ Raw hypotheses (batch of 10–20)
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │  KPI SCORER (deterministic, not LLM)                    │
  │  Score_i = w1·IC_prior + w2·breadth + w3·orthogonality  │
  │            - w4·data_cost - w5·complexity_penalty       │
  │  → Ranked shortlist: top-5 submitted to human gate #1   │
  └─────────────────────────────────────────────────────────┘
```

### LLM Prompt Engineering for Ideation

The system prompt template enforces structure and prevents hallucination drift:

```
IDEATION SYSTEM PROMPT (abbreviated):

  You are a systematic macro alpha researcher at a quant hedge fund.
  Your task: generate ONE signal hypothesis per response.

  REQUIRED OUTPUT FORMAT:
  1. ECONOMIC MECHANISM: One paragraph. No statistical language.
     Explain WHY prices should be predictable from first principles.
  2. MATHEMATICAL FORMULATION: Signal construction equation.
  3. ORTHOGONALITY CLAIM: Show why this is NOT captured by
     [carry, momentum, trend, value] with specific argument.
  4. FALSIFICATION CRITERION: What result would DISPROVE this?
     State the null hypothesis explicitly.
  5. DATA REQUIREMENTS: List every data series needed.
     Mark each as [AVAILABLE], [PURCHASABLE], or [UNAVAILABLE].
  6. KNOWN RISKS: At least 3 specific failure modes.

  FORBIDDEN: Do not propose a signal based solely on a
  historical pattern. Every signal must have an economic
  mechanism that existed BEFORE the data was collected.
```

### KPI Scoring Function

$$\text{Score}_i = \frac{w_1 \cdot \hat{IC}_i + w_2 \cdot \sqrt{B_i} + w_3 \cdot (1 - |\rho_i^{\max}|)}{1 + w_4 \cdot C_i^{\text{data}} + w_5 \cdot K_i^{\text{params}}}$$

where:
- $\hat{IC}_i$ = prior estimate from analogous signals in knowledge base
- $B_i$ = estimated breadth (assets × frequency)
- $|\rho_i^{\max}|$ = maximum correlation with existing factor library
- $C_i^{\text{data}}$ = data acquisition cost (0–1 scale)
- $K_i^{\text{params}}$ = parameter count (complexity penalty)

*Non-mathematically: score a new idea on how often we expect it to be right,
how many independent bets it gives us, and how different it is from what we
already trade — divided by how expensive and complicated it is.*

---

## §3 · Module B — Automated Feature Engineering & Signal Construction

[🔝 Back to Top](#-table-of-contents)

### The Leakage-Safe Feature Pipeline

The most dangerous failure mode in automated feature engineering is **look-ahead
contamination** — using future information to construct features that appear
predictive but are not. The pipeline enforces a strict temporal contract:

```
LEAKAGE-SAFE TEMPORAL ARCHITECTURE

  Raw data timeline:
  ───●────────●────────●────────●────────●──► t
    t-3      t-2      t-1      t        t+1

  Feature construction at time t:
  ALLOWED:   f(data[t], data[t-1], ..., data[t-L])   ← look-back
  FORBIDDEN: f(data[t+1], ..., data[t+k])             ← look-ahead

  Enforcement:
  1. All features tagged with (feature_end_time, data_end_time)
  2. feature_end_time ≤ decision_time strictly enforced by contract
  3. Unit tests: inject future value into feature → assert IC ≈ 0
     (if IC > 0.10 on shuffled future data → ALERT: leakage detected)
```

### Automated Feature Transforms

For each raw data series $x_{i,t}$, the engine auto-generates a feature library:

| Transform Family | Examples | Economic Justification |
|-----------------|---------|----------------------|
| Rolling moments | $\mu_L,\ \sigma_L,\ \text{skew}_L$ | Regime characterisation |
| Cross-sectional rank | $\Phi^{-1}(\text{rank}_t / (N+1))$ | Robust to outliers |
| Basis / spread | $(x_i - x_j) / \sigma_{i-j}$ | Relative value |
| Velocity | $\Delta_k x_i = x_{i,t} - x_{i,t-k}$ | Momentum / mean-reversion |
| Autocorrelation | $\hat{\rho}_k(x_i)$ | Persistence measure |
| Nonlinear | $\text{sign}(x)\cdot\|x\|^\alpha,\ \alpha \in \{0.5, 1, 2\}$ | Concavity control |
| Interaction | $z_i \cdot z_j \cdot \mathbf{1}[\text{regime}_t = k]$ | Conditional features |

The search space is constrained by domain rules — e.g., basis features only
between commodities in the same supply chain cluster, to prevent spurious
cross-market interactions.

### IC Pre-Screening Gate

Before any feature reaches the ML model, it passes a fast IC gate:

$$\widehat{IC}_f = \frac{1}{T}\sum_{t=1}^{T} \rho_s(f_{t}, r_{t+1})$$

Gate: $|\widehat{IC}_f| > 0.02$ AND $p$-value $< 0.10$ (Newey-West corrected).

*This eliminates ~80% of candidate features before expensive model training.*

---

## §4 · Module C — ML Validation & Anti-Overfitting Stack

[🔝 Back to Top](#-table-of-contents)

### 🗣️ Feynman Version
> "With enough parameters, any ML model can memorise historical data perfectly
> and look brilliant on paper — but fail completely on new data. We need to
> simulate, as faithfully as possible, what the model will encounter in the
> future it has never seen. Three layers of defence: honest train/test splits,
> statistical corrections for the fact that we tested many models, and a
> 'too-good-to-be-true' alarm that fires when results seem implausible."

### Layer 1 — CPCV Walk-Forward OOS

Combinatorial Purged Cross-Validation with $k=10$ folds, $p=2$ test groups:

$$\binom{10}{2} = 45 \text{ OOS paths, each with purge gap } L = 21 \text{ days}$$

The purge gap prevents serial-correlation leakage between train and test periods.
Each of the 45 paths yields an independent Sharpe estimate; the distribution
characterises the strategy's true uncertainty, not a single heroic in-sample number.

```
CPCV SCHEMATIC (10 folds, 2 test at a time — one path illustrated)

  Fold:  1    2    3    4    5    6    7    8    9    10
         ■■■  ■■■  ■■■  ■■■  ░░░  ■■■  ■■■  ■■■  ░░░  ■■■
                              ^^^                   ^^^
                        Test fold 5             Test fold 9
                        ←── Purge gap (21d) on each side ──→

  ■ = Train   ░ = Test   Gap = Purge (no data used in either)

  45 paths total → SR distribution → DSR adjusted Sharpe
```

### Layer 2 — Model Selection & Complexity Control

The framework evaluates models in increasing complexity order, stopping when
complexity no longer yields statistically significant OOS improvement:

```
MODEL LADDER (ordered by complexity):

  1. Linear (Ridge/Lasso)          ← baseline; interpretable
  2. Piecewise linear (decision stump ensemble)
  3. Gradient-boosted trees (XGBoost / LightGBM)
  4. Shallow MLP (2-layer, dropout)
  5. Temporal attention (Transformer encoder, seq_len=60)

  Selection criterion:
  Use the simplest model M_k such that:
    IC_OOS(M_{k+1}) - IC_OOS(M_k) > critical_value(α=0.05, T)

  i.e., do NOT add complexity unless it produces a statistically
  significant OOS improvement. This is the Occam's Razor gate.
```

### Layer 3 — Deflated Sharpe Ratio (DSR) with ML Trial Correction

When $M$ models are evaluated in a search, the expected maximum Sharpe of a
noise process is:

$$E\!\left[\max_{m \leq M} SR_m\right] \approx \sqrt{\frac{2 \ln M}{T}}$$

The DSR (Bailey & Lopez de Prado 2014) corrects for this:

$$DSR = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right)$$

where $SR^* = \sqrt{\frac{2 \ln M}{T}} \cdot \left(1 - \frac{\hat{\gamma}_3}{\sqrt{8\ln M}} + \hat{\gamma}_4 \cdot \frac{\ln \ln M}{8 \ln M}\right)$

*Non-mathematically: the DSR asks "given how many models we tried, what is the
probability that this Sharpe ratio is genuine and not just the lucky winner of
a noisy competition?" A DSR below 0.95 means we do not trust it.*

For the automated pipeline, $M$ is automatically tracked across the entire
search session — not just the final reported model. This is the key discipline:
the AI cannot "forget" the models it tried before finding the winner.

### Layer 4 — Regime Stress Testing

Every candidate signal is replayed through three stress regimes:

| Regime | Dates | Primary Mechanism |
|--------|-------|------------------|
| GFC | Aug 2008 – Mar 2009 | Liquidity collapse; correlation spike |
| COVID | Feb 2020 – Apr 2020 | Volatility explosion; macro surprise regime |
| Rate Shock | Jan 2022 – Oct 2022 | Carry and trend inversion; basis blow-up |

Gate:

$$
\text{Max DD}_{\text{stress}} < 2 \times \text{Max DD}_{\text{normal}}
$$

Any signal that doubles its drawdown in a stress regime relative to normal
conditions is flagged for regime-conditional weighting or retirement.

---

## §5 · Module D — Multi-Agent Research Orchestration

[🔝 Back to Top](#-table-of-contents)

### 🗣️ Feynman Version
> "Instead of one AI that does everything — which tends to agree with itself
> and miss its own errors — we use a team of specialised AI agents that
> argue with each other. One agent's job is specifically to try to break
> every signal it sees. Another checks the maths. Another writes the plain-
> English summary. A coordinator decides when they've reached consensus —
> and when they haven't, it calls a human."

### Agent Architecture

```
MULTI-AGENT ORCHESTRATION (LangGraph / AutoGen pattern)

  ┌────────────────────────────────────────────────────────────┐
  │                    COORDINATOR AGENT                        │
  │  Manages task graph, resolves conflicts, escalates to human │
  └───────┬─────────────┬───────────────┬──────────────────────┘
          │             │               │
          ▼             ▼               ▼
  ┌──────────┐  ┌──────────────┐  ┌────────────┐
  │  CRITIC  │  │  STATS AGENT │  │ RISK AGENT │
  │  AGENT   │  │              │  │            │
  │          │  │ Independently│  │ Tail-risk  │
  │ Attacks  │  │ reruns CPCV  │  │ sizing;    │
  │ hypothesis│  │ + DSR on     │  │ 2008/2020  │
  │ finds    │  │ its own data │  │ scenario   │
  │ holes    │  │ pull         │  │ replay     │
  └──────────┘  └──────────────┘  └────────────┘
          │             │               │
          └─────────────┴───────────────┘
                        │ Consolidated findings
                        ▼
              ┌──────────────────┐
              │ NARRATIVE AGENT  │
              │                  │
              │ Drafts research  │
              │ memo in plain    │
              │ English + LaTeX  │
              │ equations        │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  HUMAN GATE #3   │
              │  (PM / Senior    │
              │   Researcher)    │
              └──────────────────┘
```

### Critic Agent Prompt Design

The Critic Agent is the most important safeguard. Its system prompt:

```
CRITIC AGENT SYSTEM PROMPT (abbreviated):

  Your ONLY job is to find weaknesses. You are NOT trying to
  help the signal succeed. For every signal presented:

  1. LOOK-AHEAD CHECK: Can this signal be constructed without
     future data? Show the exact timestamp logic.
  2. OVERFITTING CHECK: How many free parameters? Is the
     Sharpe consistent with DSR > 0.95?
  3. REGIME CHECK: Does IC hold in 2008, 2020, and 2022?
     Show the IC time-series, not just the average.
  4. COST CHECK: Does the gross alpha survive realistic TCA?
     Compute TC at 2× estimated spread + 50% slippage.
  5. ORTHOGONALITY CHECK: Run regression vs. [carry, trend,
     momentum] and report the residual R².

  Output: A structured list of PASS / WARN / FAIL for each
  check, with quantitative evidence. Never say "looks good"
  unless all 5 checks are explicit PASS.
```

### Conflict Resolution Protocol

```
COORDINATOR DECISION TREE

  All agents PASS?
  ├── YES → Advance to human gate #3
  └── NO  → Check severity:
            ├── Any FAIL (hard)? → REJECT → back to Module A
            ├── ≥ 2 WARNs?       → escalate to human immediately
            └── 1 WARN?          → Critic + Stats agents re-run
                                   with stricter parameters
                                   ├── Resolves? → Advance
                                   └── Persists? → Escalate
```

---

## §6 · Module E — Human Oversight & Hallucination Guardrails

[🔝 Back to Top](#-table-of-contents)

### Why AI Alone Is Insufficient

LLMs and ML models suffer from three failure modes in quant research:

| Failure Mode | Manifestation in Quant Research | Mitigation |
|-------------|--------------------------------|-----------|
| **Hallucination** | LLM cites non-existent paper; fabricates IC statistic | Fact-checker agent verifies every numeric claim against raw data |
| **Overfitting** | ML model finds spurious pattern in backtest | DSR + CPCV + M-trial correction (Layer 3 above) |
| **Distribution shift** | Model trained on 2010–2020 fails in 2022 rate shock | Mandatory stress test on held-out crisis regimes |

### The Three Human Gates

```
HUMAN GATE DECISION FRAMEWORK

  GATE #1 — Hypothesis Review (Researcher level)
  ├── Input:  Top-5 AI-scored hypotheses + economic narratives
  ├── Review: ~30 min; checks economic plausibility
  ├── Output: Approve / Modify / Reject each hypothesis
  └── Guardrail: Must reject at least 1 of 5 (forces critical thinking)

  GATE #2 — Statistical Scorecard Review (Senior Researcher)
  ├── Input:  CPCV paths, DSR, IC time-series, regime breakdown
  ├── Review: ~2 hours; focuses on regime sensitivity and DSR
  ├── Output: Approve for agent review / Request re-run / Reject
  └── Guardrail: DSR < 0.95 is a hard reject; no exceptions

  GATE #3 — PM Approval (Portfolio Manager)
  ├── Input:  Narrative memo from Narrative Agent + conflict log
  ├── Review: ~1 hour; focuses on portfolio fit and sizing
  ├── Output: Go / No-Go / Conditional (staging with γ = 0.30)
  └── Guardrail: New signal must not increase portfolio VaR > 5%
```

### Hallucination Detection Architecture

```
HALLUCINATION DETECTOR (runs on all LLM outputs)

  For every NUMERIC CLAIM in LLM output:
  ┌──────────────────────────────────────────────────────────┐
  │  1. Extract claim: "IC = 0.047 for ISCF on copper"       │
  │  2. Query data store: pull ISCF IC series for copper      │
  │  3. Compute actual value: IC = 0.043                      │
  │  4. Tolerance check: |0.047 - 0.043| / 0.043 = 9.3%      │
  │     < 15% tolerance → PASS                               │
  │     ≥ 15% tolerance → FLAG as hallucination              │
  └──────────────────────────────────────────────────────────┘

  For every CITATION in LLM output:
  ┌──────────────────────────────────────────────────────────┐
  │  1. Extract: "Grinold (1989) shows IR = IC * sqrt(N)"    │
  │  2. Verify paper exists in knowledge base                 │
  │  3. Verify equation matches stored version               │
  │  4. FAIL → replace with [CITATION UNVERIFIED] + flag     │
  └──────────────────────────────────────────────────────────┘

  "Too Good to Be True" alarm:
  ├── Annual Sharpe > 3.0 in OOS → FLAG
  ├── IC > 0.10 consistently     → FLAG
  ├── Max DD < 2% over 5 years   → FLAG
  └── All flags → mandatory human review before proceeding
```

### Immutable Audit Trail

Every AI decision is logged with:

```python
AuditEntry = {
    "timestamp":     ISO-8601,
    "module":        "A" | "B" | "C" | "D" | "E",
    "agent":         "ideation" | "critic" | "stats" | "risk" | "narrative",
    "input_hash":    SHA-256(input_data),
    "output_hash":   SHA-256(output),
    "data_sources":  [list of data series used with timestamps],
    "human_gate":    null | {gate: 1|2|3, reviewer: str, decision: str},
    "model_trials":  M,   # cumulative trial count for DSR correction
}
```

This makes the pipeline fully auditable: any production signal can be traced
back to the exact data, model, and human decision that created it.

---

## §7 · ISCF & MGD as Test Bench

[🔝 Back to Top](#-table-of-contents)

### How ISCF Maps to the AI Factory

```
ISCF THROUGH THE AI FACTORY PIPELINE

  MODULE A (Ideation):
  ├── Hypothesis: "Physical delivery constraints create basis premiums
  │   invisible to roll-yield carry signals"
  ├── Economic mechanism: Theory of Storage (Working 1949) → non-linear
  │   convenience yield → hockey-stick backwardation
  └── KPI score: IC_prior=0.04, breadth=2016, ρ_carry=0.08 → Score=0.71

  MODULE B (Feature Engineering):
  ├── Auto-generated: basis/vol z-score, LME inventory velocity,
  │   freight rate change, seasonal adjustment
  ├── Macro-beta filter: (1 - β_macro) suppression
  └── Gram-Schmidt orthogonalisation vs. [trend, mom, carry]

  MODULE C (Validation):
  ├── CPCV: 45 paths, SR distribution μ=1.41, σ=0.31
  ├── DSR: 0.97 (M=2 trial correction) → PASS
  └── Stress test: Max DD ratio 2008=1.7× (PASS, < 2× gate)

  MODULE D (Multi-agent):
  ├── Critic: "LME inventory proxy via EIA is 48h lagged" → WARN
  ├── Stats:  Reruns confirm IC=0.043, ICIR=1.38 → PASS
  ├── Risk:   ADV cap 10% sufficient at $50M AUM → PASS
  └── Resolution: WARN on data lag → mitigated by EMA smoothing

  MODULE E (Human Gate):
  ├── Gate #2: DSR=0.97 approved; proxy data risk noted
  └── Gate #3: γ=0.30 staging deployment (proxy data regime)
```

### How MGD Maps to the AI Factory

```
MGD THROUGH THE AI FACTORY PIPELINE

  MODULE A (Ideation):
  ├── Hypothesis: "FX forward curves embed growth expectations;
  │   macro data releases create exploitable repricing lags"
  ├── Economic mechanism: Muth (1961) rational expectations with
  │   institutional flow lag → 1-3 period price adjustment delay
  └── KPI score: IC_prior=0.038, breadth=2016, ρ_carry=0.06 → Score=0.68

  MODULE B (Feature Engineering):
  ├── Auto-generated: composite surprise, EMA(τ=5) divergence,
  │   Kalman-filtered nowcast residual, cross-pair divergence spread
  └── Gram-Schmidt vs. existing FX factors

  MODULE C (Validation):
  ├── CPCV: 45 paths, SR distribution μ=1.33, σ=0.28
  ├── DSR: 0.96 → PASS
  └── Stress test: 2020 IC halved during COVID surprise regime → WARN

  MODULE D (Multi-agent):
  ├── Critic: "q=0.29 Kalman signal-to-noise breaks in 2020" → WARN
  ├── Stats:  Confirms Kalman degrades; adaptive q improves: ICIR 1.38→1.52
  ├── Risk:   Regime-conditional weighting reduces 2020 DD by 35%
  └── Resolution: Adaptive q upgrade → WARN resolved

  MODULE E (Human Gate):
  └── Gate #3: Approved with adaptive-q Kalman as mandatory upgrade
```

---

## §8 · Generalisation: Making the Framework Signal-Agnostic

[🔝 Back to Top](#-table-of-contents)

### The Signal Protocol Interface

The factory is made generic through a typed **Signal Protocol** that any signal
must implement:

```python
class SignalProtocol:
    """
    Abstract contract every signal must satisfy.
    The AI factory dispatches through this interface.
    """

    # ── Required metadata ──────────────────────────────────────────
    signal_id:       str          # unique identifier
    universe:        list[str]    # asset list
    frequency:       str          # "4x_daily" | "daily" | "weekly"
    factor_family:   FactorFamily # CARRY | MOMENTUM | TREND | VALUE | OTHER

    # ── Required data declaration ──────────────────────────────────
    def required_data(self) -> list[DataContract]:
        """Declare data sources BEFORE any computation."""
        ...

    # ── Required signal construction ───────────────────────────────
    def compute(self, data: DataSnapshot) -> SignalResult:
        """
        Compute signal from data snapshot.
        data.timestamp is the decision time.
        data contains ONLY data available at decision time.
        """
        ...

    # ── Required falsification criterion ───────────────────────────
    def null_hypothesis(self) -> str:
        """State what would DISPROVE this signal."""
        ...

    # ── Required KPI targets ───────────────────────────────────────
    def kpi_targets(self) -> KPITargets:
        """Declare IC target, IR hurdle, max DD, half-life floor."""
        ...
```

### Taxonomy of Signal Types the Factory Can Handle

```
GENERIC SIGNAL TAXONOMY

  Dimension 1 — Data Type:
  ├── Market data (price, volume, OI)
  ├── Macro data (economic releases, central bank)
  ├── Alternative data (satellite, shipping, sentiment)
  └── Derived data (volatility surface, order flow)

  Dimension 2 — Mechanism:
  ├── Mean-reversion   (OU process; inventory/spread normalisation)
  ├── Momentum         (trend continuation; flow persistence)
  ├── Carry            (yield advantage; cost of carry)
  ├── Value            (fundamental anchor; long-run mean)
  └── Catalyst         (event-driven; macro announcement)

  Dimension 3 — Frequency:
  ├── Intraday         (tick to 30-min; microstructure)
  ├── 4× daily         (session-level; ISCF / MGD regime)
  ├── Daily            (EOD rebalance; macro factors)
  └── Weekly/Monthly   (slow macro; value / carry)

  The factory applies the SAME 5-module pipeline regardless of
  which (data, mechanism, frequency) cell the signal occupies.
  Only the feature transform library and IC pre-screening window
  are adapted per frequency.
```

### Scaling the Factory: Parallel Signal Search

With the generic interface, the factory runs multiple hypothesis streams in parallel:

$$\text{Throughput} = \frac{N_{\text{hypotheses}} \times N_{\text{agents}}}{T_{\text{CPCV}} / N_{\text{compute-nodes}}}$$

At 4 compute nodes and 2-hour CPCV runs, the factory evaluates ~10 signals per day —
versus 1 signal per week manually. The **bottleneck shifts from compute to human
review capacity**, which is why Gates #1–3 are deliberately lightweight (30 min / 2h / 1h).

---

## §9 · Implementation Roadmap: Months 1–6

[🔝 Back to Top](#-table-of-contents)

```
MONTH 1 — FOUNDATION
══════════════════════════════════════════════════════════════════════════
  Week 1-2: Build Signal Protocol interface + adapter for ISCF & MGD
  Week 3-4: Stand up knowledge base (RAG on quant corpus, 200 papers)
  Deliverable: ISCF & MGD running through generic pipeline end-to-end
  Human gate: Researcher validates pipeline output matches manual results

MONTH 2 — IDEATION ENGINE
══════════════════════════════════════════════════════════════════════════
  Week 1-2: Fine-tune LLM on quant finance corpus (LoRA, base: Llama-3)
  Week 3:   Build KPI scorer + hallucination detector
  Week 4:   First run: generate 20 hypotheses; human selects top 5
  Deliverable: 5 candidate hypotheses with economic narratives + data plans
  KPI: At least 3 of 5 pass Critic Agent review

MONTH 3 — FEATURE ENGINEERING + ML VALIDATION
══════════════════════════════════════════════════════════════════════════
  Week 1-2: Feature pipeline (tsfresh integration + domain constraints)
  Week 2-3: CPCV + DSR harness with M-trial tracking
  Week 4:   Model ladder (Ridge → XGBoost → shallow Transformer)
  Deliverable: Automated scorecard for all 5 hypotheses
  KPI: DSR > 0.95 for ≥ 2 signals; regime stress passed

MONTH 4 — MULTI-AGENT ORCHESTRATION
══════════════════════════════════════════════════════════════════════════
  Week 1-2: Critic, Stats, Risk, Narrative agents (LangGraph)
  Week 3:   Coordinator + conflict resolution protocol
  Week 4:   Full pipeline dry run on ISCF & MGD (known-good answers)
  Deliverable: Multi-agent consensus memo for 2 new signals
  KPI: Agent findings match manual review on ISCF / MGD test cases

MONTH 5 — INTEGRATION + HUMAN GATES
══════════════════════════════════════════════════════════════════════════
  Week 1-2: Audit trail + Gate #1/2/3 workflow (Slack + Notion)
  Week 3:   Staged deployment of best new signal at γ = 0.30
  Week 4:   Fill rate audit; TCA model calibration
  Deliverable: First AI-ideated signal in paper trading with live KPI monitoring
  KPI: IC OOS consistent with backtest within 1 ICIR standard deviation

MONTH 6 — PRODUCTION + ITERATION
══════════════════════════════════════════════════════════════════════════
  Week 1-2: γ → 0.95 if IC holds; second signal enters staging
  Week 3-4: Retrospective: where did AI save time? Where did it mislead?
  Deliverable: Full pipeline operational; second iteration of ideation begun
  KPI: ≥ 2 AI-ideated signals live; ideation-to-staging < 5 weeks
```

---

## §10 · Key Equations & Complexity Analysis

[🔝 Back to Top](#-table-of-contents)

### Ideation Pipeline

| Equation | Formula | Intuition |
|----------|---------|-----------|
| IC decay with crowding | $`\dot{IC}(t) = -\kappa \cdot IC(t) - \delta \cdot C(t) \cdot IC(t)`$ | Speed-to-market value. The loss of signal ($`\dot{IC}`$) is proportional to its current strength. $`\kappa`$ is the natural, organic decay (e.g., macro data becoming old news), and $`\delta \cdot C(t)`$ represents the premium decay forced by market crowding (other funds running the same model). |
| KPI score | $`\text{Score} = \left( w_1 \hat{IC} + w_2\sqrt{B} - w_3 \|\rho^{\max}\| \right) \cdot e^{-(w_4 C^{\text{data}} + w_5 K)}`$ | Multi-objective signal ranking. If a signal uses massive alternative data costs ($`C^{\text{data}}`$) or explodes in complexity ($`K`$), its score is smoothly squeezed toward zero. |
| IC pre-screen | $`\\|\widehat{IC}_f\\| > 0.02, \quad p < 0.10`$ | Fast feature filter. **$`\\|\widehat{IC}_f\\| > 0.02`$:** For a systematic macro signal (trading highly liquid indices, FX, or commodities), an absolute Information Coefficient above $`0.02`$ ($`2\\%`$) over a meaningful horizon is a standard threshold for "economically viable" predictive power. **$`p < 0.10`$:** The p-value check ensures that this $`0.02`$ IC isn't just statistical noise, establishing a $`90\\%`$ confidence interval that the correlation is real. |

### ML Validation

| Equation | Formula | Intuition |
|----------|---------|-----------|
| Expected max SR (noise) | $`E[\max_{m \leq M} SR_m] \approx \sqrt{2\ln M / T}`$ | Selection bias benchmark |
| DSR | $`SR^* = \sqrt{\sigma^2_{SR}} \left( (1-\gamma)\Phi^{-1}\left(1-\frac{1}{M}\right) + \gamma\Phi^{-1}\left(1-\frac{1}{M}e^{-1}\right) \right)`$ | Selection-bias-corrected SR |
| Occam gate | $`IC_{\text{OOS}}(M_{k+1}) - IC_{\text{OOS}}(M_k) > z_{1-\alpha} \cdot \frac{\sigma_{\Delta IC}}{\sqrt{T_{\text{eff}}}}`$ | Model complexity cutoff. Instead of checking if the new model is just "slightly better," this formula creates a **statistical volatility channel**. If the difference in performance ($`\Delta IC`$) is highly volatile across different macro regimes, $`\sigma_{\Delta IC}`$ expands, making the hurdle rate significantly harder to cross. This prevents the pipeline from adopting a complex model that performs phenomenally well in an expansionary regime but collapses during a liquidity crisis. In practice, the out-of-sample vectors $`IC_{\text{OOS}}(M_{k+1})$ and $IC_{\text{OOS}}(M_k)`$ should not be derived from a single historical split. They are generated via **Combinatorial Purged and Embargoed Cross-Validation (CPCV)**. By testing the models across multiple combinatorial paths that have been strictly purged of lookahead bias and embargoed to prevent overlapping data leakage, you generate a highly stable, non-overfitted distribution of $\Delta IC$. The Occam Gate evaluates the cross-validated paths simultaneously to ensure the complex model introduces persistent structural alpha rather than localized noise-fitting. |
| Throughput | $`\text{Throughput} = N_{\text{hyp}} \times N_{\text{agents}} / (T_{\text{CPCV}} / N_{\text{nodes}}) \quad \text{or } \text{Throughput} = \frac{N_{\text{hyp}} \times N_{\text{agents}} \times N_{\text{nodes}}}{T_{\text{CPCV}}}`$  | Factory scaling equation. This is a fantastic operational engineering equation. It quantifies your compute efficiency when running massive ML search spaces. This cleanly shows that your research velocity scales linearly with your compute cluster size ($`N_{\text{nodes}}`$), but is throttled by how strictly you structure your validation metrics ($`T_{\text{CPCV}}`$). If you increase the number of paths in your cross-validation to prevent overfitting, your throughput drops unless you scale up your cluster hardware. |

### Guardrails

| Check | Trigger | Action |
|-------|---------|--------|
| Hallucination | Numeric deviation > 15% vs. raw data | FLAG + human review |
| Too-good-to-be-true | SR > 3.0 OOS or IC > 0.10 | Mandatory human gate |
| Regime fragility | Max DD stress > 2× normal | Regime-conditional weight or reject |
| Trial inflation | $M > 20$ without DSR recalculation | Recompute $SR^*$ with actual $M$ |

---

```
SUMMARY: THE AI ALPHA FACTORY IN ONE DIAGRAM

  Human                AI                    Data
  ─────                ──                    ────
  Gate #1              Ideation Engine   ←── Knowledge Base (RAG)
    │                      │
    │                  Feature Pipeline  ←── Market + Alt Data
    │                      │
  Gate #2              CPCV + DSR Stack
    │                      │
    │                  Multi-Agent Team
    │                  (Critic / Stats /
    │                   Risk / Narrative)
    │                      │
  Gate #3              Hallucination
    │                  Detector
    │                      │
    └──────────────── Go / No-Go ──────► Production Signal
                                         (γ-gated, audited)

  "AI provides speed and breadth.
   Human provides judgment and accountability.
   The combination is strictly superior to either alone."
```

---

*Shaikat Majumdar · HLS Trading Interview Preparation · June 2026*
*AI + ML Next Frontier: Compressing the Ideation-to-P&L Cycle*
