# Deep Analysis: Why We Max Out at 66% and How to Fix It

## The Raw Data (Empirical, Not Theoretical)

I ran a diagnostic across 500 live BTC 15-minute candles. Here is exactly what the scorer produces:

```
Probability Distribution:
   25%-29% :    4
   30%-34% :   15
   35%-39% :   30
   40%-44% :   63
   45%-49% :   99   ← bulk of candles
   50%-54% :  152   ← peak: most candles score ~50%
   55%-59% :  103
   60%-64% :   32
   65%-69% :    2   ← only TWO candles ever reached 65%
   70%+    :    0   ← literally impossible with current math

Mean: 50.3%   Max ever reached: 66.0%
```

**The distribution is a tight bell curve centered at exactly 50%.** This is the mathematical fingerprint of a system where opposing signals cancel each other out.

---

## The 5 Structural Problems (Root Causes)

### Problem 1: Trend and Momentum Are Structurally Contradictory

This is the single biggest issue. At the exact moment you want to enter a trade (a reversal), the Trend and Momentum indicators fight each other:

**Example — Classic oversold bounce setup (the most profitable pattern in crypto):**

| Signal | What happens | Score | Weight | Contribution |
|:---|:---|:---|:---|:---|
| SMA200 | Price below SMA200 in downtrend | -1.0 | 10% | **-10.0** |
| SMA50 | Price below SMA50 | -0.8 | 8% | **-6.4** |
| MACD | Histogram negative | -0.5 | 10% | **-5.0** |
| RSI | RSI=28 oversold — BUY! | +1.0 | 12% | **+12.0** |
| StochRSI | K=15 oversold — BUY! | +0.7 | 10% | **+7.0** |

**Net from just these 5: -2.4** → The trend kills the momentum signal.

This means the most common profitable setup in crypto (oversold bounce) is *penalized* by the system.

### Problem 2: SR Zone (Weight 18%) Is Dead on Fast Timeframes

The S/R proximity signal has the **highest weight in the entire system (18 points)**, but it almost never fires on 15m candles.

The code uses `zone_pct = 0.3%` to determine "at support". But on 15-minute BTC candles, a single candle can move 0.3-0.5%. By the time the candle closes, the price has already bounced away from the support level.

**Actual average contribution: +0.579 out of a possible +18.0.** That means 96.8% of this signal's potential is wasted.

### Problem 3: Volume Is Structurally Capped

```python
if ratio >= 2.0: return 0.4   # Maximum possible return
```

Volume has weight 8, but its max raw score is 0.4 → **max contribution = 3.2 out of 8.0 possible.** 

There are 4.8 "phantom points" that can *never* be earned. The system acts as if Volume could contribute 8 points, but it physically cannot contribute more than 3.2.

### Problem 4: VWAP Penalizes Pullback Entries

A pullback to VWAP in an uptrend is a classic buy opportunity. But the scorer gives it `-0.3` because price is below VWAP — **actively penalizing the setup you want to buy**.

### Problem 5: The Linear Average Architecture

All 10 signals vote independently in a flat weighted average. This is fundamentally wrong because:
- Trend and Momentum are **anti-correlated at entries** (by design)
- Volume should be a **multiplier**, not an additive score
- S/R is a **gate** (are you at the right place?), not a voter

---

## The Fix: 3-Layer Gated Architecture

Instead of one flat weighted average, restructure as three independent gates:

### Gate A — Location (0-40 points): "Are we at the right price?"
| Signal | Max Points | Logic |
|:---|:---|:---|
| Near SR Support (any TF) | 20 | At support → 20, within 0.5% → 15, within 1% → 8 |
| Bollinger Band lower zone | 12 | bb_pct < 0.2 → 12, < 0.4 → 6 |
| Pivot S1/S2 zone | 8 | At S1 → 5, at S2 → 8 |

### Gate B — Confirmation (0-35 points): "Is momentum confirming?"
| Signal | Max Points | Logic |
|:---|:---|:---|
| RSI oversold reversal | 15 | RSI < 30 → 15, < 40 → 8 |
| StochRSI K crossing D from below 20 | 10 | Cross + oversold → 10 |
| MACD histogram turning positive | 10 | Negative → positive flip → 10, just positive → 5 |

### Gate C — Context (0-25 points): "Is the bigger picture helping?"
| Signal | Max Points | Logic |
|:---|:---|:---|
| SMA200 trend alignment | 10 | Price > SMA200 → 10, below → 0 (NOT negative) |
| Volume confirmation | 8 | Ratio > 1.5 → 8, > 1.0 → 4, < 0.7 → 0 |
| VWAP alignment | 7 | Above VWAP → 7, pullback to VWAP in uptrend → 5 |

**Total possible: 100 points**

### Key Architectural Difference

- **Trend is a BONUS, not a penalty.** Being in an uptrend adds +10. Being in a downtrend adds +0. It never subtracts.
- **S/R is the primary driver.** You need to be at the right price first.
- **Volume is confirmatory.** It amplifies but never overrides.
- **MACD direction change matters more than absolute level.**

### Expected Score Distribution with New Architecture

| Setup Type | Current Score | New Score |
|:---|:---|:---|
| Oversold bounce at support in uptrend | 58% | **85-90%** |
| Oversold bounce at support in downtrend | 45% (killed by trend) | **60-75%** (still valid) |
| Random mid-range candle, no setup | 50% | **15-25%** (properly filtered) |
| Strong trend continuation with volume | 55% | **70-80%** |

The new system would produce a **bimodal distribution** (low scores for noise, high scores for setups) instead of the current bell curve at 50%.

---

## Alternative Approach: Regime-Based Multi-Strategy

Instead of one scorer for all market conditions, detect the **regime** first, then apply the right sub-strategy:

1. **Trending Market** (ADX > 25): Use trend-following signals (EMA crossover, MACD direction). Ignore S/R.
2. **Ranging Market** (ADX < 20): Use mean-reversion signals (RSI oversold, BB bounce, S/R). Ignore trend.
3. **Volatile Breakout** (BB squeeze + volume spike): Use breakout signals. Wider SL.

This is how professional quant firms operate — they don't use one model for all conditions.

---

## My Recommendation

**Do both, in two phases:**

1. **Phase 1 (immediate): Rewrite the scorer** with the 3-Layer Gated architecture. This fixes the mathematical ceiling and makes the probability number meaningful. One file change (`signal_scorer.py`), ~150 lines.

2. **Phase 2 (next session): Add regime detection.** Compute ADX (Average Directional Index) as a new indicator. Use it to switch between trend-following and mean-reversion strategies automatically.

**Phase 1 alone should move the score distribution from `max 66%` to `max 85-90%`**, giving you plenty of high-confidence trades at the 70% threshold you want.
