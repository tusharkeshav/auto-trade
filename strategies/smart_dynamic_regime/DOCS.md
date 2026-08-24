# 🧠 Strategy 5: Smart Dynamic Macro-Regime Capital Allocator

## 1. Executive Overview
- **Philosophy**: An institutional tactical asset allocation engine that eliminates the "static 50/50 drag" by dynamically shifting capital to where the alpha is highest:
  - **In Trending Bull Markets**: 60% Sector ETF Momentum + 40% Large-Cap Stocks.
  - **In Choppy Sideways Consolidations**: 70% Large-Cap Pullbacks (60d RS Filter) + 30% Gold (`GOLDBEES`).
  - **In Severe Bear Market Crashes**: 100% Sovereign Gold / Liquid Cash.

---

## 2. Mathematical Decision Logic

```
1. IF NIFTY <= SMA200 OR EMA12 < EMA50 * 0.99:
     REGIME = BEAR_DEFENSE
     ALLOCATION = 100% GOLDBEES.NS (Safe Haven Gold)

2. ELSE IF ADX(14) >= 22.0 AND EMA12 > EMA50:
     REGIME = TRENDING_BULL
     ALLOCATION = 60% Sector ETF Momentum + 40% Large-Cap Stocks

3. ELSE:
     REGIME = CHOPPY_RANGE
     ALLOCATION = 70% Large-Cap Pullbacks + 30% GOLDBEES.NS (Gold)
```

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/smart_dynamic_regime/run_test.py
```
