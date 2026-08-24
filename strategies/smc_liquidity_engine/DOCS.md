# 🏦 Strategy 6: Smart Money Concepts (SMC) Quantitative Engine

## 1. Executive Overview
- **Philosophy**: Institutional price-action model tracking **Market Structure Breaks (BOS)**, **Fair Value Gaps (FVG)**, and **Order Blocks (OB)**. Only buys when price retraces into the **50% Fibonacci Discount Zone**, achieving high asymmetric risk-to-reward ratios ($1:4+$).
- **Candidate Universe**: 8 Secular Blue-Chips (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `LT.NS`, `BHARTIARTL.NS`, `SBIN.NS`, `SUNPHARMA.NS`, `NTPC.NS`).
- **Benchmark**: `^NSEI` (NIFTY 50 Index).

---

## 2. Mathematical Formulations & Rules

### 🛡️ 1. Macro Trend Shield
$$\text{Macro Condition} = \text{Close}_{\text{NIFTY}} > \text{SMA}_{200}(\text{NIFTY})$$

### 📐 2. Market Structure & 50% Fibonacci Discount Gate
1. **Swing High / Low**: 20-bar rolling highs and lows.
2. **Equilibrium Formula**:
   $$\text{Equilibrium} = \frac{\text{Swing High} + \text{Swing Low}}{2}$$
3. **Discount Rule**: Only long when $\text{Price}_t < \text{Equilibrium}$ (Never buy at Premium!).

### ⚡ 3. Bullish Fair Value Gap (FVG) / Imbalance
In a 3-candle sequence:
$$\text{Low}(C_t) > \text{High}(C_{t-2})$$
*(Price is drawn to mitigate this gap before resuming its uptrend).*

### 🎯 4. Asymmetric Risk & Target
- **Stop Loss**: Placed below the Order Block / Swing Low: $\text{Entry} - 1.25 \times \text{ATR}_{14}$ (~2.0% risk).
- **Take Profit**: Targeted at the previous Swing High: $\text{Entry} + \max(\text{Swing High} - \text{Entry}, 4.0 \times \text{ATR})$ ($1:4+$ R:R).
- **Trailing Profit Lock ($+0.5R$)**: Trailed to $\text{Entry} + 0.5 \times \text{SL Distance}$ when profit reaches $+2.0R$.

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/smc_liquidity_engine/run_test.py
```
