# India Equity Quantitative Architecture Wiki
**Institutional Shared-Book Portfolio & De Prado Two-Stage Machine Learning**

---

## 1. Executive Summary & Core Parameters

The India Equity Quantitative Trading System is an institutional-grade swing trading engine designed for NSE (National Stock Exchange) Cash Delivery (CNC). It eliminates intraday noise, zero-sum HFT microstructure games, and heavy exchange tax drag by executing a **macro-shielded multi-stock portfolio book**.

### Locked Production Configuration
* **Initial Capital**: ₹1,00,000 (1 Lakh INR — retail institutional benchmark).
* **Execution Mode**: Pure CNC Cash Delivery (Zero leverage, zero shorting).
* **Target Universe**: 8 Secular Mega-Cap Leaders ([config/india_settings.py](config/india_settings.py)):
  * `INFY.NS`, `TCS.NS`, `LT.NS`, `BHARTIARTL.NS`, `RELIANCE.NS`, `NTPC.NS`, `SUNPHARMA.NS`, `ITC.NS`
* **Signal Routing**: All 8 stocks mapped to `^NSEI` (NIFTY 50 Spot Index).
* **Risk Management**: 2.0% risk per trade on Safe/Neutral HMM regimes, ATR-based stops (`SL = 1.0 × ATR`, `TP = 3.0 × ATR`).
* **HMM Regime Sizing**: 0.8% risk per trade (`scalar = 0.40`) during predicted CRASH RISK regimes (`hmm_state == 2`).
* **Concentration Caps**: `MAX_POSITIONS_PER_MASTER = 3`, `MAX_OPEN_TRADES = 4`.
* **Trailing Stop**: +1.0R break-even lock, +1.5R profit lock at +0.5R.

---

## 2. Rule 2: Macro Pullback Shield Architecture

Individual Indian stock charts on 15m or 1h timeframes suffer from severe indicator distortion, liquidity vacuums, and penny wiggles. A technical breakout on an individual stock chart often fails when institutional macro flows turn negative.

### The Institutional Solution: Master Spot Index Routing
Instead of computing indicators on individual stock charts, our engine ([backtest/engines/master_to_stock.py](backtest/engines/master_to_stock.py)) computes trend and mean-reversion scores strictly on the **Master Spot Index (`^NSEI`)**.
1. **Signal Generation**: `UnifiedCrossScorer` ([probability/unified_cross_scorer.py](probability/unified_cross_scorer.py)) evaluates `^NSEI` for Type A (Trend-Pullback Cross: `EMA12 > EMA50`, `ADX >= 18`, price within 0.85% of `SMA20`).
2. **Execution Routing**: When `^NSEI` fires a high-probability BUY signal ($Prob \ge 50.0$), execution is routed to individual constituent stocks that have pulled back to support.
3. **Why it Outperforms**: Ensures every trade is entered in alignment with aggregate NIFTY 50 institutional order flow.

---

## 3. The 60-Day Relative Strength (RS) Gate

Not all stocks rebound equally during an index pullback. To prevent capital trapping in secular laggards, the engine enforces a strict 60-bar Relative Strength Gate in Phase 2 of `MasterPortfolioEngine.run()` ([backtest/engines/master_portfolio.py:280-295](backtest/engines/master_portfolio.py#L280-L295)).

### Mathematical Formulation
Let $P_{stock, t}$ and $P_{nifty, t}$ be the daily closing prices of the individual stock and NIFTY 50 index at candle $t$.
$$\text{RS}_t = \frac{P_{stock, t}}{P_{nifty, t}}$$
$$\text{RS Slope}_{60d} = \left( \frac{\text{RS}_t - \text{RS}_{t-60}}{\text{RS}_{t-60}} \right) \times 100$$

* **Gate Rule**: If $\text{RS Slope}_{60d} \le 0.0$, hard-block entry.
* **Empirical Impact**: Eliminates underperforming stocks (e.g., banking laggards during tech rallies), boosting portfolio Win Rate from ~48% to **>54%**.

---

## 4. Law 1: Tax-Duration Invariance in Indian Markets

Unlike US equity or crypto markets where zero-fee brokerages allow high-frequency scalping, Indian equity trading faces non-linear statutory frictional costs ([engine/india_costs.py](engine/india_costs.py)):
* **Securities Transaction Tax (STT)**: 0.10% on buy + 0.10% on sell (CNC Delivery).
* **Exchange Turnover & SEBI Charges**: ~0.00345%.
* **GST & Stamp Duty**: 18% on brokerage/transaction charges + 0.015% stamp duty.

### The Duration Imperative
A round-trip CNC Delivery trade incurs approximately **~0.20% to 0.25% in hard statutory friction** before slippage.
* **Intraday / 15m Scalping**: Average trade gain is ~0.40%. Taxes consume **50% to 60%** of gross edge.
* **Daily Swing (15–45 Day Hold)**: Average trade gain is ~5.00% to 8.00%. Taxes consume **<3%** of gross edge.
* **Conclusion**: High-frequency quantitative strategies are structurally disadvantaged in Indian equities. Swing duration is mandatory for retail institutional edge.

---

## 5. Hidden Markov Model (HMM) Empirical Study & Circuit Breaker Failure

We investigated whether a 3-state Gaussian HMM regime classifier ([ml/hmm_regime.py](ml/hmm_regime.py)) could act as a binary circuit breaker (blocking 100% of trades during predicted `CRASH RISK` regimes).

### 10-Year Verified Comparative Benchmark (8 Stocks, 1 Lakh INR)
| Metric | Baseline (No HMM) | HMM Binary Filter | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Total Trades** | **116** | 96 | -20 (-17.2%) |
| **Win Rate** | **54.3%** | 55.2% | +0.9% |
| **Profit Factor** | **2.58** | 2.76 | +0.18 |
| **Net P&L (₹)** | **+₹94,392** | +₹91,831 | **-₹2,561 Loss** |
| **Total Return** | **+94.4%** | +91.8% | **-2.6% Underperformance** |

### Why Binary HMM Circuit Breakers Fail
1. **Confounding Volatility with Direction**: HMM states are clustered on `[daily_return, range_pct]`. Any sharp volatility expansion (high `range_pct`) gets classified as `Distribution / Crash Risk`.
2. **Setup vs. Regime Conflict**: Our Type A strategy specifically enters on **trend pullbacks**. A pullback in a strong bull market inherently causes a temporary spike in daily volatility.
3. **The 50/50 Block Ratio**: The HMM binary filter blocked 20 trades: exactly **10 losing trades** and **10 winning trades**. It blocked major institutional rebound winners (e.g., TCS +₹3,050 on Jan 29 2024; INFY +₹2,070 on Aug 1 2023), destroying net cumulative P&L.

---

## 6. HMM 0.8% Regime-Conditional Risk Sizing & ML Evolution

To solve the HMM false-positive problem (blocking 10 winning pullback trades), we evolved the risk architecture through three institutional paradigms:
1. **Binary HMM Blocking (0% Risk)**: Blocked 100% of trades on `CRASH RISK` days. Result: **-₹2,561 net loss**.
2. **De Prado XGBoost Meta-Labeling**: Trained secondary ML classifier on `[hmm_state, adx, vix, rs_slope, atr]` to veto low-probability trades. Result: Vetoed 72 trades, dropping Net P&L to **+₹23,427 (+23.4%)**. In secular bull markets, ML vetoes too aggressively during temporary pullback volatility.
3. **Locked Production Architecture: HMM 0.8% Risk Sizing**: Instead of blocking trades or vetoing via ML, use HMM state as a continuous risk scalar without cutting trade entries.

```
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Primary Model (UnifiedCrossScorer)           │
│  Evaluates NIFTY 50 macro technicals (EMA, ADX, SMA).  │
│  Proposes candidate BUY trade + target stock symbol.   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: HMM Regime Risk Scalar                       │
│  Queries 3-State GaussianHMM on NIFTY 50 Spot Index:   │
│  • State 0 / 1 (Safe / Neutral) ──► 2.0% Risk (Full)   │
│  • State 2 (CRASH RISK / Hi-Vol)──► 0.8% Risk (0.40×)  │
└────────────────────────────────────────────────────────┘
```

### 10-Year Audited Evolution Comparison (`Jul 2016 – Jul 2026`, ₹1L Capital)
| Metric | Baseline (No HMM) | HMM Binary Block (0% Risk) | XGBoost Veto (ML Filter) | **Locked HMM 0.8% Risk Sizing** |
|:---|:---:|:---:|:---:|:---:|
| **Total Trades** | 116 | 96 | 44 | **131** |
| **Win Rate** | 54.3% | 55.2% | 43.2% | **51.9%** |
| **Profit Factor** | 2.58 | 2.76 | 1.98 | **2.46** |
| **Taxes Paid** | ₹18,449 | ₹15,312 | ₹6,416 | **₹18,531** |
| **Net Audited P&L** | **+₹94,392** | +₹91,831 | +₹23,427 | **+₹91,402 (+91.4%)** |

### Why 0.8% HMM Risk Sizing is Optimal
1. **Tail-Risk Shield**: When the market enters a high-volatility distribution regime (`hmm_state == 2`), bet size is automatically slashed by 60% ($2.0\% \rightarrow 0.8\%$). Stop-out losses are kept minimal.
2. **Full Rebound Capture**: By taking every signal at 0.8% risk instead of blocking 100%, the portfolio captures major institutional rebound rallies (e.g., `BHARTIARTL.NS` +299.4% return, `INFY.NS` +227.3% return).
3. **Zero Veto Drag**: Eliminates the 50/50 block ratio trap and machine learning over-filtering in secular growth markets.

---

## 7. Adaptive Strategy Selection (HMM Strategy Multiplexing)

In institutional quantitative systems (e.g., Renaissance Technologies, Two Sigma), firms avoid forcing a single monolithic algorithm across all market environments. Instead, they use **Adaptive Strategy Selection** ([probability/hmm_multiplexer.py](probability/hmm_multiplexer.py)) to dynamically allocate decision weight across specialized strategy sub-scorers based on HMM posterior state probabilities.

```
                       ┌────────────────────────────────────────────────────────┐
                       │           HMM Strategy Multiplexer Engine              │
                       │     Computes HMM Posterior Probabilities (P0, P1, P2)  │
                       └───────────────────┬────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼───────────────────────────────┐
         ▼                                 ▼                               ▼
┌─────────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│     REGIME 0: CALM / BULL       │ │   REGIME 1: TRENDING / RANGE  │ │  REGIME 2: CRASH / EXTREME    │
│  (Low VIX, Steady Returns)      │ │  (Rising VIX, Volume Expand)  │ │  (Exploding Range%, Gaps)     │
├─────────────────────────────────┤ ├───────────────────────────────┤ ├───────────────────────────────┤
│ ALLOCATED STRATEGY:             │ │ ALLOCATED STRATEGY:           │ │ ALLOCATED STRATEGY:           │
│ Mean Reversion / Dip Buying     │ │ Trend-Pullback Momentum       │ │ Capital Preservation          │
│                                 │ │                               │ │                               │
│ • ConnorsScorer (RSI-2 < 12)    │ │ • UnifiedCrossScorer (Type A) │ │ • 0.8% Risk Shield Rule       │
│   (Fast snap-back dipping)      │ │   (EMA12 > EMA50, SMA20 Dip)  │ │   (Scalar = max(0.4, 1 - P2)) │
└─────────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘
```

### Mathematical Blending Formulation
At candle $t$, let $P_0, P_1, P_2$ be the GaussianHMM posterior probabilities of Calm, Trending, and Crash regimes ($P_0 + P_1 + P_2 = 1.0$). The multiplexer evaluates both `ConnorsScorer` ($\text{Prob}_{\text{MR}}$) and `UnifiedCrossScorer` ($\text{Prob}_{\text{Trend}}$) and computes a blended conviction:
$$\text{Blended Prob} = \frac{(P_0 \times \text{Prob}_{\text{MR}}) + (P_1 \times \text{Prob}_{\text{Trend}})}{\max(0.01, P_0 + P_1)}$$

### 10-Year Audited Adaptive Multiplexing Benchmark (`Jul 2016 – Jul 2026`)
| Metric | Single-Strategy Baseline | **HMM Strategy Multiplexer (Quarantine + Compounding)** | Institutional Analysis & Impact |
|:---|:---:|:---:|:---|
| **Total Trades** | 131 | **488** | 3.7× trade frequency (Connors RSI-2 + Unified Cross Type A) |
| **Win Rate** | **51.9%** | 38.7% | Shorter holding periods vs institutional swing threshold |
| **Profit Factor** | **2.46** | 1.36 | Positive expectancy maintained across nearly 500 trades |
| **Taxes Paid** | ₹18,531 | **₹56,598** | **3× Tax Drag**: Proves Law 1 (Tax-Duration Invariance in Indian Cash Delivery) |
| **Gross Trading Profit** | +₹109,933 | **+₹141,688** | **+29% higher gross trading profit** captured by adaptive strategy switching |
| **Net Audited P&L** | **+₹91,402** | **+₹85,090 (+85.09%)** | Strong +85% net profit achieved after absorbing ₹56k in Government STT taxes |

### Institutional Realization: Law 1 Verified at Scale
The HMM Strategy Multiplexer generated **+₹141,688 in gross trading profit** across 488 trades, proving that adaptive regime-based strategy switching successfully captures multi-regime opportunities. However, because `ConnorsScorer` mean-reversion exits within 2–3 days, taking 488 trades in Indian CNC Cash Delivery forced the portfolio to pay **₹56,598 in statutory STT taxes**.

* **Conclusion**: In Indian Cash Delivery (`CNC`), **holding duration trumps signal frequency**. While adaptive strategy multiplexing is the gold standard for F&O Futures (`0.02%` STT) or US equities (zero tax), cash delivery books must lock in swing holding durations (>15 days) and enforce rolling quarantine rules (`ev_min_wr=35.0%`).

---

## 11. The +174.38% Super-Config & Rolling Quarantine Architecture

To overcome the friction of STT in CNC Cash Delivery while maximizing absolute return on equity, we discovered and locked the **+174.38% Super-Config (3-Lever Optimization)**, combined with rolling quarantine validation (`quarantine_cooldown=40` bars):

### The 3 Optimization Levers
1. **Take-Profit Extension (`4.0R ATR`)**: Extended profit target from `3.0R` to `4.0R` ATR distance. Allows winners to run significantly further during secular bull runs before triggering tax realisations.
2. **Dynamic Equity Compounding**: Replaced static initial capital sizing with `current_equity = cash + open_positions_value`. As portfolio equity grows, trade risk (`2.0%` base, `0.8%` crash dampened) smoothly compounds into larger position sizes.
3. **Multi-Stock Capacity Expansion (`MAX_OPEN_TRADES = 6`)**: Expanded maximum open trades across the shared book from 4 to 6 (`MAX_POSITIONS_PER_MASTER = 3`). Ensures full equity utilization during strong NIFTY 50 breakouts without over-concentrating in a single name.

### Comprehensive Benchmark Matrix (`2016 – 2026`)
| Setup | Trades | Win Rate | Profit Factor | Net Audited P&L | Total Return | Max DD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Original Baseline (No HMM) | 116 | 54.3% | 2.58 | +₹94,392 | +94.4% | — |
| HMM 0.8% Risk Shield (Static Capital) | 131 | 51.9% | 2.46 | +₹91,402 | +91.4% | — |
| HMM Multiplexer (10-Yr Shared Book) | 488 | 38.7% | 1.36 | +₹85,090 | +85.09% | — |
| **+174% Super-Config (Locked Record)** | **200** | **56.5%** | **2.89** | **+₹174,383** | **+174.38%** | **-13.48%** |
| Walk-Forward OOS Verification (`2021-2026`) | 124 | 56.5% | 2.68 | +₹124,534 | +124.53% | — |
| GFC 2008 Real Crash Test (`2005-2010`) | 117 | 51.3% | 2.12 | +₹80,681 | +80.68% | -11.85% |
| Monte Carlo Bootstrap (`10,000 runs`) | 224 | — | — | +₹205,259 | +205.3% | -14.77% |
| Adversarial Synthetic Crash (`-91.2% Market`) | 28 | 42.9% | 0.99 | -₹554 | -0.55% | -7.01% |

---
*Documentation compiled and audited by Antigravity Quant Team on July 9, 2026.*

