# Intraday Cross-Sectional Alpha Research Roadmap

## 1. Project objective

Build a **new, standalone intraday systematic-equity research project from scratch**. It is separate from A3/B050; only research principles carry over: PIT universes, chronological validation, leakage controls, trial logging, bootstrap confidence intervals, realistic transaction costs, and strict separation of exploratory and confirmatory analysis.

Central question:

> **Does a statistically robust and economically exploitable cross-sectional signal exist in liquid U.S. equities at intraday horizons of roughly 30–120 minutes?**

This is **not HFT or minute-level scalping**. The intended strategy operates at a small number of fixed decision times per trading day.

## 2. Core formulation

For stock $i$ at decision time $t$:

$$
X_{i,t} \rightarrow E[R_{i,t:t+h}-R^{SPY}_{t:t+h}\mid X_{i,t}]
$$

with the primary target horizon **frozen now**, before any exploration:

$$h=60\text{ minutes}.$$

Horizons $H\in\{30,120\}$ minutes are permitted only as **diagnostic cross-checks** during Phase 0. They cannot be promoted to the primary target after the fact — changing the primary horizon starts a new experiment against a new independent dataset.

The model is primarily a **cross-sectional ranker**, not a market-direction predictor. Market timing/exposure is a separate later risk-layer problem.

**Scope v1 is long-only.** Shorting is out of scope for this project version. All portfolio construction, benchmarks and gates are defined for a long-only book.

## 3. Benchmarks and metrics

### Investment benchmark

**SPY buy-and-hold** over the same calendar period. Portfolio-level comparison includes CAGR, Sharpe, **Calmar**, Maximum Drawdown, annualized volatility, net P&L and turnover.

The realized portfolio is a **long-only Top-N** book. The primary economic benchmark is Top-N excess return over SPY on the same intraday interval:

$$
R^{excess}_{portfolio,t}=R^{TopN}_{t:t+60m}-R^{SPY}_{t:t+60m}.
$$

$$
Calmar=\frac{CAGR}{|MaxDrawdown|}
$$

### Signal benchmark

Use SPY over exactly the same intraday interval:

$$
y_{i,t}=R_{i,t:t+60m}-R^{SPY}_{t:t+60m}.
$$

Do not use daily SPY close-to-close return as the target benchmark for an intraday prediction.

### Naive strategy baselines

Before ML, compare against simple rules: equal-weight liquid universe, short-horizon momentum, short-horizon reversal, and sector-relative momentum if Phase 0 supports it.

Research hierarchy:

$$SPY \rightarrow Naive\ Signal \rightarrow Linear\ Model \rightarrow LightGBM.$$

---

# Phase 0 — Hypothesis Discovery

## Goal

Determine whether exploitable intraday cross-sectional structure appears to exist **before building a predictive model**.

No portfolio optimizer, production broker integration, complex ML or hyperparameter search belongs here.

$$\boxed{\text{Is there a sufficiently promising intraday signal worth testing properly?}}$$

Phase 0 is **hypothesis discovery only**, not a stability assessment. Yahoo's ~60-day 5-minute window is insufficient for confident inference about the persistence of any effect through time. True stability, statistical significance and economic magnitude are assessed only on the independent Phase 2 dataset.

Its dataset is subsequently considered contaminated for confirmatory inference.

## Phase 0.1 — Prototype dataset

Start with inexpensive/free intraday data, initially Yahoo Finance if adequate.

Target:

- approximately 300–500 liquid U.S. equities
- SPY
- relevant sector ETFs
- 5-minute OHLCV
- maximum available intraday history

Required raw fields:

```text
timestamp
ticker
open
high
low
close
volume
```

Normalize timestamps to `America/New_York`. Trading logic must not depend on the researcher's local timezone.

For this exploratory stage, a current liquid universe is acceptable because Phase 0 is used only for hypothesis generation, not for estimating historical strategy performance.

## Phase 0.2 — Data-quality audit

For every ticker/date calculate:

$$
Coverage_{i,d}=\frac{ObservedBars_{i,d}}{ExpectedBars_d}.
$$

Audit missing/duplicate bars and sessions, zero-volume bars, halts, early closes, timestamp alignment, DST, splits, extreme discontinuities, and synchronization with SPY/sector ETFs.

**Never forward-fill missing OHLCV through absent trading intervals.** Produce a coverage report before signal research begins.

## Phase 0A — Intraday return map

Test basic momentum/reversal structure without ML.

The search space below is **preregistered**: it is fixed at the start of Phase 0 and no lookback, horizon or decision time may be added mid-exploration.

Preregistered past-return windows:

$$L\in\{5,15,30,60,120\}\text{ minutes}.$$

Primary future horizon (frozen by Section 2):

$$H=60\text{ minutes}.$$

Diagnostic-only future horizons (never promoted to primary):

$$H^{diag}\in\{30,120\}\text{ minutes}.$$

Preregistered decision-time grid on the ET clock:

$$T^{decision}\in\{10{:}00,11{:}00,12{:}00,13{:}00,14{:}00,15{:}00\}\text{ ET}.$$

For each decision timestamp calculate cross-sectional Spearman rank IC:

$$
IC_{L,H,t}=Spearman_{CS}(R_{i,t-L:t},R_{i,t:t+H}).
$$

Interpretation:

$$IC>0\Rightarrow momentum/continuation$$

$$IC<0\Rightarrow reversal.$$

Create lookback × horizon heatmaps and repeat over the decision-time grid above. The purpose is to identify broad structure, not select the single best cell. The final decision-time schedule is chosen at the end of Phase 0 from this preregistered grid and is not modified afterward.

## Phase 0B — Market- and sector-relative signals

Repeat after removing broad market movement.

Simple market-relative return:

$$R_i^{rel}=R_i-R_{SPY}.$$

Beta-adjusted residual:

$$R_i^{res}=R_i-\beta_{i,M}R_M.$$

Sector-adjusted residual:

$$R_i^{res}=R_i-\beta_{i,M}R_M-\beta_{i,S}R_S.$$

Compare raw, SPY-relative, beta-adjusted and sector-adjusted signals with the same IC methodology.

## Phase 0C — Volume, volatility and state conditioning

Investigate whether momentum/reversal depends on market state.

Example volume surprise:

$$
VolumeSurprise_{i,t}=\frac{Volume_{i,t-L:t}}{MedianHistoricalVolume_{i,\ same\ clock\ interval}}.
$$

Other candidate state variables: realized volatility, high-low range, abnormal range, distance from daily open, opening gap, distance from VWAP, SPY return, sector return, breadth and time of day.

Example conditional test:

$$IC(R_{30},R_{+60}\mid VolumeSurprise>Q_{80})$$

versus

$$IC(R_{30},R_{+60}\mid VolumeSurprise<Q_{20}).$$

Avoid uncontrolled combinatorial bucket searches.

## Phase 0D — Cross-sectional monotonicity

For promising signals, sort each timestamp's cross-section into deciles:

$$Q_1,\ldots,Q_{10}.$$

Measure subsequent returns by bucket and especially:

$$Spread^{diag}=R_{Q10}-R_{Q1}.$$

Prefer broadly monotonic behavior, although perfect monotonicity is not required. At this stage the focus is gross signal economics, not final strategy profitability.

The $Q10-Q1$ spread is a **diagnostic of ranking quality only** — it is never realized as a portfolio. Scope v1 is long-only, so the realized-portfolio equivalent of this diagnostic is $R_{Q10}-R^{SPY}$, i.e. how much extra return the top decile earns over the market on the same interval.

## Phase 0E — Weak stability preview

Check the candidate effect across the weeks, days, time-of-day buckets, weekdays, sectors, volatility regimes and positive/negative market regimes that happen to be present in the Yahoo window.

This is a **weak preview**, not a stability confirmation. With ~60 days of 5-minute data, any bucket-level claim (e.g. "the effect works in low-vol regime") is a hypothesis to be tested in Phase 2, not a finding.

Use this stage as a red-flag filter: if the effect is dominated by one week, one time-of-day bucket or one sector, downgrade or drop it. If it looks broadly consistent, it earns the right to be confirmatory-tested, not the label of "confirmed."

## Phase 0F — Hypothesis freeze

At the end of Phase 0 select **exactly one primary hypothesis**. It receives **exactly one** independent confirmatory test in Phase 2. There is no "iterate the hypothesis on the confirmatory set."

Example only:

> 30-minute market/sector residual momentum conditional on abnormal volume predicts the next 60-minute cross-sectional excess return.

Freeze signal definition, lookback $L$, decision-time schedule (chosen from the preregistered Phase 0A grid), conditioning variables, expected direction, universe rules and core evaluation metrics. The target horizon $H=60$m is already frozen from Section 2 and cannot be reopened here. The benchmark is Top-N excess over SPY, long-only.

Record the exact start and end dates of the Phase 0 Yahoo window. These dates constitute the **contamination window** and must be embargoed from the Phase 2 confirmatory test.

The Phase 0 Yahoo dataset is now **exploratory/in-sample** and cannot be presented as independent confirmation.

---

# Phase 1 — Proper Historical Dataset

Acquire research-grade independent intraday history, preferably roughly 2018→present or the longest practical high-quality period.

Prefer storing **1-minute source data**, even if features use 5/15/30/60-minute aggregation.

Desired data: 1-minute OHLCV, historical symbols/security IDs, corporate actions, delisted securities where possible, and bid/ask or BBO if economically feasible.

**Contamination embargo.** The exact Phase 0 Yahoo window is contamination. Remove it entirely from the Phase 2 partitions (train / validation / WFOOS / locked test). No confirmatory metric may be computed on those dates. Document the embargoed date range in the trial registry alongside the frozen hypothesis.

## Phase 1.1 — Point-in-time universe

Use:

$$Universe(t)$$

not today's membership retrospectively. Correctly handle historical membership, ticker changes, mergers, delistings where available and corporate actions. Survivorship bias is unacceptable from this phase onward.

---

# Phase 2 — Confirmatory Signal Test

Test the **frozen Phase 0 hypothesis** on independent long-history data. Do not search for a replacement hypothesis if it fails.

Use chronological partitions:

```text
Train
Validation
WFOOS
Locked Test
```

Apply purge/embargo appropriate to the prediction horizon.

Primary diagnostics: mean/median cross-sectional Rank IC, confidence interval, temporally appropriate t-statistic, stability through time, decile monotonicity, top-bottom spread and regime stability. Use block bootstrap where appropriate rather than treating overlapping observations as independent.

### Phase 2 gate

Proceed only if the signal retains the expected sign, has economically meaningful magnitude, is reasonably stable, is not driven by a tiny number of periods, and survives independent data.

If it fails:

$$\boxed{STOP}$$

Do not mine the confirmatory dataset for another specification.

---

# Phase 3 — Predictive Models

Only after the underlying signal survives confirmation.

## 3.1 Linear baseline

Start with Ridge; optionally Elastic Net.

## 3.2 LightGBM

Train LightGBM on the same frozen target and chronological framework.

Initial feature families should remain compact: 5/15/30/60-minute returns, relative/residual returns, cross-sectional ranks, volume surprise, realized volatility, high-low range, opening gap, distance from open/VWAP, market return, sector return, breadth and time-of-day variables.

Do not start with hundreds of TA indicators, signatures, neural networks, transformers or arbitrary feature generation.

Key incremental test:

$$\Delta IC_{ML-Baseline}>0.$$

Use paired/bootstrap confidence intervals for model comparisons.

---

# Phase 4 — Execution and Cost Model

Intraday economics require explicit execution modelling. Never assume:

$$FillPrice=SignalClose.$$

If BBO exists:

$$BuyPrice\approx Ask+Slippage$$

$$SellPrice\approx Bid-Slippage.$$

Introduce realistic execution delay, e.g.:

$$Signal_t\rightarrow Order\rightarrow Fill_{t+1m}.$$

Include bid-ask spread, slippage, broker commissions, regulatory/exchange fees, FX costs where applicable, and relevant rejected/partial fills.

Stress-test additional execution costs such as 1, 2, 5 and 10 bp.

Report decomposition:

$$GrossEdge\rightarrow Spread\rightarrow Slippage\rightarrow Fees\rightarrow NetEdge.$$

**NetEdge as a function of deployed capital.** Capital is not fixed a priori. Compute

$$NetEdge(C)=GrossEdge-Costs(C)$$

where $Costs(C)$ includes fixed-per-trade fees (dominant at small $C$), spread and ADV-scaled market-impact (dominant at large $C$). Identify the **minimum viable capital** $C^{*}$ such that

$$NetEdge(C)>0\ \ \forall C\in[C^{*},\ C^{**}],$$

with $C^{**}$ the capacity ceiling where impact erases the edge. A strategy that exists only under optimistic execution assumptions or only at capital levels outside a realistic operating band fails.

---

# Phase 5 — Simple Portfolio

**Long-only Top-N**, equal-weight. No complex optimizer initially. No short leg. The $Q10-Q1$ diagnostic from Phase 0D is never realized as a portfolio — long-short is a separate research cycle, not part of v1.

$$Portfolio=Top10\rightarrow EqualWeight$$

or

$$Portfolio=Top20\rightarrow EqualWeight.$$

Measure CAGR, Sharpe, **Calmar**, Maximum Drawdown, annualized volatility, net P&L, turnover, hit rate, market beta, sector exposure, gross/net exposure, average holding period, trades/day and cost as a fraction of gross edge.

Compare directly with SPY buy-and-hold over the same period and with frozen naive intraday baselines. The primary economic gate is Top-N excess return over SPY on the same intraday intervals, net of realistic costs.

---

# Phase 6 — Holding and Rebalancing Policy

Only after the simple portfolio demonstrates net economic value.

Candidate policies: fixed 60-minute holding period, retain while rank remains sufficiently high, no-trade band, minimum expected-edge threshold.

Example:

$$Rank_{i,t+60}>threshold\Rightarrow Hold.$$

Objective: reduce turnover without destroying signal exposure. Any new policy must be evaluated with an appropriate independent/nested research process.

---

# Phase 7 — Risk and Portfolio Layer

Only after stock-selection alpha and execution economics are established.

Scope v1 remains **long-only**. A long-short overlay is a future research cycle, not part of v1's risk layer.

Potential constraints: maximum position weight, sector exposure, market beta, gross exposure, volatility, turnover, liquidity/ADV and maximum concurrent positions.

Potential market-state exposure:

$$GrossExposure_t=f(MarketState_t).$$

Market timing remains conceptually separate from stock-selection quality. Do not let a market-direction overlay hide a weak cross-sectional model.

---

# Phase 8 — Paper Trading

Freeze the complete system before paper trading:

```text
Live market data
        ↓
feature calculation
        ↓
prediction
        ↓
ranking
        ↓
target portfolio
        ↓
orders
        ↓
broker API
        ↓
fill reconciliation
        ↓
positions / P&L / logs
```

IBKR is a primary production candidate, but keep execution broker-agnostic, e.g. adapters for IBKR and Trading 212.

Paper trading is primarily an **engineering/execution validation stage**, not another model-selection dataset.

Audit data arrival timing, feature reproducibility, order timing, rejected/partial/duplicate orders, API outages, position reconciliation, expected vs simulated fills, P&L reconciliation and restart/recovery behavior.

---

# Phase 9 — Tiny Live

After stable paper operation, deploy small real capital.

Primary objective: measure real execution rather than maximize return.

$$ExecutionError=ActualFill-ExpectedFill.$$

Compare expected vs realized spread, slippage, transaction costs and turnover. Recalculate historical economics using empirically observed live execution costs. If real execution materially invalidates backtest assumptions, do not scale.

---

# Phase 10 — Scale and Further Research

Only after a meaningful untouched forward period should capital increase.

Subsequent research cycles may investigate alternative horizons such as 30m/120m, additional independent signals, improved execution, dynamic holding periods, market-state models, portfolio optimization, alternative universes or richer quote/order-book features.

These are **new research hypotheses**, not post-hoc tuning. Each gets its own preregistered specification and validation budget.

---

# Research Controls

## Leakage

Every feature must satisfy:

$$FeatureTime\le DecisionTime.$$

Explicitly test bar-boundary leakage, VWAP leakage, daily-high/low leakage, volume normalization using future same-day data, corporate-action leakage, universe-membership leakage and timestamp/DST errors.

## Multiple testing

Phase 0 is explicitly exploratory. Once the hypothesis is frozen, Phase 1/2 becomes confirmatory.

Phase 0's search space is the **preregistered** grid: lookbacks $L\in\{5,15,30,60,120\}$m, primary horizon $H=60$m (with $H\in\{30,120\}$m diagnostic-only), the fixed ET decision-time grid, the fixed list of conditioning variables (volume surprise, realized vol, HL range, distance from open/VWAP, opening gap, SPY/sector state, breadth, time of day), and the raw/SPY-relative/beta-adjusted/sector-adjusted return specifications. No new axis may be added mid-exploration.

**No Bonferroni or family-wise correction is applied inside Phase 0.** Phase 0 makes no inferential claims — significance is not asserted on the exploratory dataset. Multiple-testing control is instead enforced at the Phase 0/1 boundary by the single-hypothesis rule: **exactly one frozen hypothesis** exits Phase 0, and it receives **exactly one** independent confirmatory test in Phase 2.

Do not repeatedly modify horizon, lookback, decision time, feature definition or conditioning rule until significance appears. Every material hypothesis change starts a new experiment on a new independent dataset.

## Trial registry

Record every experiment, preferably JSONL or equivalent:

```text
trial_id
timestamp
git_commit
dataset_version
universe_version
train_period
validation_period
target
horizon
decision_times
feature_set
model
hyperparameters
cost_model
metrics
result
decision
```

Failed experiments remain in the registry.

## Statistical inference

Prefer cross-sectional Spearman Rank IC, paired comparisons, block bootstrap, confidence intervals, stability plots and year/regime decomposition.

Do not treat hundreds of stocks at one timestamp as hundreds of independent temporal observations. The economically relevant independent information is heavily constrained by the number of trading periods/days.

---

# Core Metrics

## Signal-level

Primary: **Rank IC**.

Supporting: mean/median IC, IC confidence interval, IC t-statistic, IC stability, decile monotonicity, top-bottom spread and coverage.

## Portfolio-level

Primary economic metrics:

$$CAGR$$

$$Sharpe$$

$$Calmar=\frac{CAGR}{|MDD|}$$

$$MaxDrawdown$$

plus annualized volatility, net P&L, turnover, hit rate, beta, sector exposure, cost drag, trades/day and average holding period.

All final portfolio metrics must be reported **after realistic costs**.

---

# Stop Rules

**Signal failure:** frozen Phase 0 hypothesis does not reproduce on independent historical data.

**ML failure:** ML provides no meaningful incremental value over the simple signal baseline.

**Economic failure:** gross signal exists but $NetEdge\le0$ after realistic execution costs.

**Stability failure:** performance is dominated by a narrow regime/year/time bucket without ex-ante justification.

**Execution failure:** observed paper/live execution costs invalidate historical assumptions.

A failed gate is a research result, not an invitation to search the same dataset for another specification.

---

# Project Progression

```text
PHASE 0
Exploratory Yahoo 5m data
        ↓
Intraday signal map
        ↓
Momentum / reversal / relative-strength structure
        ↓
Volume & volatility conditioning
        ↓
Stability + monotonicity
        ↓
ONE HYPOTHESIS FROZEN
        │
        ▼
PHASE 1
Independent long-history intraday dataset
+ PIT universe
        │
        ▼
PHASE 2
Confirmatory test
        │
        ├── FAIL → STOP
        │
        ▼
PHASE 3
Ridge → LightGBM
        │
        ▼
PHASE 4
Spread + slippage + fees + execution delay
        │
        ├── NET EDGE ≤ 0 → STOP
        │
        ▼
PHASE 5
Simple Top-N portfolio
Sharpe + Calmar + MDD + CAGR
        │
        ▼
PHASE 6
Holding / turnover policy
        │
        ▼
PHASE 7
Risk / portfolio layer
        │
        ▼
PHASE 8
Paper trading
        │
        ▼
PHASE 9
Tiny live
        │
        ▼
PHASE 10
Scale / new research cycle
```

---

# Initial Implementation Target

The first implementation milestone is **Phase 0 only**. Do not build the production model or broker integration yet.

Initial deliverables:

1. Yahoo 5-minute downloader with caching and reproducibility.
2. Universe definition.
3. SPY and sector ETF data.
4. Data-quality and coverage audit.
5. Intraday return/IC maps.
6. Market-relative and sector-relative signal maps.
7. Volume/volatility conditioning diagnostics.
8. Decile monotonicity diagnostics.
9. Weak stability preview (Phase 0E).
10. Phase 0 research report.
11. Exactly one frozen hypothesis for independent testing, if evidence warrants one.

If no sufficiently convincing hypothesis emerges, Phase 0 concludes with **STOP** rather than progressing automatically to ML.

---

# Guiding Principle

$$
\boxed{
Data\rightarrow Market\ Structure\rightarrow Hypothesis\rightarrow Independent\ Confirmation\rightarrow ML\rightarrow Execution\rightarrow Portfolio\rightarrow Paper\rightarrow Live
}
$$

Optimize for **falsifiability and honest out-of-sample evidence**, not for producing a profitable-looking backtest.
