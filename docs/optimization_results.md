# AI Crypto Trading Bot: Optimization & Strategy Documentation

This document records the core trading strategy logic and tracks the progression of our parameter optimizations.

## 1. Core Trading Strategy

The bot operates on a **Walk-Forward Probability Matrix**. It does not rely on simple crossovers; instead, it uses a multi-factor scorer (Trend + Momentum + Volatility + Volume) to generate a composite "Probability Score" (0-100%).

### Entry Logic
- **LONG:** Enters when Probability Score ≥ `Threshold` on candle close.
- **SHORT:** (Pending calibration, planned for ≤ 35%).
- **Filter:** Will not enter a new trade if currently in a position for the same asset.

### Risk Management & Position Sizing
- **Risk Budget:** 2% of total capital risked per trade.
- **Stop Loss (SL):** Dynamic, based on Average True Range (ATR) multiplied by the `ATR_Mult`.
- **Position Size:** Calculated so that if the SL is hit, the portfolio loses exactly 2%.

### Exit Lifecycle (Lifecycle Engine)
- **Take Profit 1 (TP1):** Set at 1:1 Risk-to-Reward. Closes 50% of the position.
- **Breakeven Stop:** Once TP1 is hit, the Stop Loss is immediately moved to the Entry Price.
- **Take Profit 2 (TP2):** Set at 1.5:1 Risk-to-Reward. Closes the remaining 50%.
- **Conflict Rule:** During backtesting, if a single candle hits both SL and TP1, the system conservatively assumes the SL was hit first to prevent lookahead bias.

---

## 2. Optimization Progression

We ran three major grid search sweeps to find the optimal balance between safety (Drawdown), profitability (Profit Factor), and frequency (Trades).

### Phase 1: The Initial Conservative Sweep
* **Goal:** Prove the baseline strategy could be profitable.
* **Data Window:** ~50 Days (300 candles on 4h).
* **Parameters Tested:** Timeframes (`1h, 2h, 4h`), Thresholds (`65% to 75%`).
* **Result:** `4h / 70% Threshold / 1.0x ATR`. 
* **Outcome:** Highly profitable (Profit Factor 2.41, +3% P&L) but severely low frequency (only 3 trades in 50 days).

### Phase 2: Expanded Historical Validation (1000 Candles)
* **Goal:** Test the conservative settings against a much longer, harsher market environment (166 days / 6 months of data).
* **Parameters Tested:** Same as Phase 1 (`1h, 2h, 4h`, `65% to 75%`).
* **Result:** `4h / 65% Threshold / 0.75x ATR`. 
* **Outcome:** The system became *too* strict. Over 6 months, the `70%` threshold was filtered out entirely because it produced fewer than 3 trades. The `65%` threshold produced exactly 3 trades (all winners, 100% WR, +7.69% P&L). 
* **Conclusion:** The AI was successfully filtering out bad chop markets, but the 4h timeframe is too slow for active trading.

### Phase 3: The High-Frequency Shift
* **Goal:** Unlock trade frequency by looking at faster markets and lowering the AI's "pickiness."
* **Parameters Tested:** Faster timeframes (`15m, 30m, 1h`), Lower Thresholds (`55% to 65%`), ATR (`0.75x to 1.50x`).
* **Result:** The 15-minute chart completely solved the frequency issue, provided we used a wider stop loss (`1.50x ATR`) to survive intra-candle noise.

#### Top Configurations Found (Phase 3):
| Focus | Timeframe | Threshold | ATR Mult | Trades | Win Rate | Profit Factor | P&L % | Max DD % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Balanced (Best)** | `15m` | `58%` | `1.50x` | 34 | 61.8% | 1.59 | +16.50% | 3.96% |
| **Max Frequency** | `15m` | `55%` | `1.25x` | 66 | 56.1% | 1.42 | +27.04% | 11.53% |
| **Max Profit** | `15m` | `55%` | `1.50x` | 45 | 60.0% | 1.51 | +20.34% | 9.20% |

---

## 3. Current System State & Next Steps

The engine, risk manager, and backtester are fully operational and capable of instant offline optimization via local Parquet/CSV caching. 

**Pending Upgrades:**
1. **SHORT Calibrations:** The system is currently LONG-only. We need to run the optimizer to find the inverse parameters (e.g., probability `≤ 35%`) to allow the bot to short the market.
2. **Multi-Timeframe Filter:** Require a `4h` bullish trend confirmation before allowing a `15m` LONG entry signal to trigger.
3. **Macro Context (LLM):** Feed daily news sentiment to an LLM to dynamically shift the `58%` threshold up or down based on global market fear/greed.
