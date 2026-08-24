# 🎯 Strategy 2: Large-Cap Pullback Engine (60-Day RS Gate)

## 1. Executive Overview
- **Philosophy**: Mean-reversion swing engine that buys institutional wholesale pullbacks at 20-day Simple Moving Average (`SMA20`) support, strictly filtered by 60-day Relative Strength against NIFTY 50.
- **Candidate Universe**: 8 Secular Blue-Chips (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `LT.NS`, `BHARTIARTL.NS`, `SBIN.NS`, `SUNPHARMA.NS`, `NTPC.NS`).
- **Benchmark**: `^NSEI` (NIFTY 50 Index).

---

## 2. Mathematical Formulations & Rules

### 🛡️ 1. The 60-Day Relative Strength (RS) Gate
A stock is strictly disqualified from buying unless its 60-day relative strength slope against NIFTY 50 is positive:
$$\text{RS}_{\text{slope}} = \frac{\frac{\text{Price}_t}{\text{NIFTY}_t} - \frac{\text{Price}_{t-60}}{\text{NIFTY}_{t-60}}}{\frac{\text{Price}_{t-60}}{\text{NIFTY}_{t-60}}} \times 100 > 0.0$$

### 🟢 2. The Wholesale Dip Setup (SMA20 Re-Test)
$$\text{Price}_{t-1} \le \text{SMA}_{20}(t-1) \times 1.005 \quad \text{AND} \quad \text{Price}_t > \text{SMA}_{20}(t)$$
$$40.0 \le \text{RSI}_{14}(t) \le 58.0$$

### 🎯 3. Risk-to-Reward & Trailing Profit Lock
- **Stop Loss**: $\text{Entry} - 1.25 \times \text{ATR}_{14}$ (~2.0% - 2.5% risk).
- **Take Profit**: $\text{Entry} + 4.00 \times \text{ATR}_{14}$ (~8.0% - 14.0% target).
- **Trailing Profit Lock ($+0.5R$)**: When price reaches $+2.0R$ floating gain, Stop Loss is trailed to **$\text{Entry} + 0.5 \times \text{SL Distance}$**, locking in guaranteed profit.

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/largecap_pullback/run_test.py
```
