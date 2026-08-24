# 🚀 Strategy 3: Minervini Volatility Contraction Pattern (VCP) Engine

## 1. Executive Overview
- **Philosophy**: Developed by 2-time US Investing Champion Mark Minervini. Buys explosive new momentum highs only after price volatility has compressed into an ultra-tight coiled spring ($T_1 \rightarrow T_2 \rightarrow T_3$), completely shaking out overhead trapped supply.
- **Candidate Universe**: 8 Secular Blue-Chips (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `LT.NS`, `BHARTIARTL.NS`, `SBIN.NS`, `SUNPHARMA.NS`, `NTPC.NS`).
- **Benchmark**: `^NSEI` (NIFTY 50 Index).

---

## 2. Mathematical Formulations & Rules

### 🛡️ 1. Trend Template (Minervini Stage 2 Uptrend)
$$\text{Price} > \text{SMA}_{200} \quad \text{AND} \quad \text{Price} > \text{SMA}_{50} \quad \text{AND} \quad \text{NIFTY} > \text{SMA}_{200}$$

### 🔍 2. Volatility Compression Metric (Bollinger Bandwidth Squeeze)
$$\text{BB Width} = \frac{\text{Upper BB}_{20, 2\sigma} - \text{Lower BB}_{20, 2\sigma}}{\text{Middle SMA}_{20}} < 0.10$$

### ⚡ 3. The Pivot Trigger & Volume Expansion
$$\text{Price}_t \ge \text{Max}(\text{High}_{t-20 \dots t-1}) \times 0.999 \quad \text{AND} \quad \text{Price}_{t-1} < \text{High}_{t-21 \dots t-2}$$
$$\text{Volume}_t \ge 1.15 \times \text{Volume MA}_{20}$$

### 🎯 4. Asymmetric Risk-to-Reward Ratio ($1:3.5+$)
- **Stop Loss**: $\text{Entry} - 1.25 \times \text{ATR}_{14}$ (~2.0% - 2.8% risk).
- **Take Profit**: $\text{Entry} + 3.50 \times \text{ATR}_{14}$ (~9.0% - 13.0% target).
- **Trailing Profit Lock ($+0.5R$)**: When price reaches $+2.0R$ gain, Stop Loss is trailed to **$\text{Entry} + 0.5 \times \text{SL Distance}$**, locking in guaranteed profit.

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/vcp_breakout/run_test.py
```
