# 📈 Strategy 1: Sector ETF Dual-Momentum Rotation Engine

## 1. Executive Overview
- **Philosophy**: Gary Antonacci's Dual-Momentum model tailored for Indian Sector ETFs. Rotates capital into the Top 2 leading economic sectors while preserving capital via a 200 SMA Macro Cash Shield.
- **Base Universe**: `NIFTYBEES.NS`, `BANKBEES.NS`, `ITBEES.NS`, `AUTOBEES.NS`, `PHARMABEES.NS`, `CPSEETF.NS`.
- **Safe Haven Asset**: `GOLDBEES.NS` (Sovereign Gold Hedge).
- **Benchmark**: `^NSEI` (NIFTY 50 Index).

---

## 2. Mathematical Formulations & Rules

### 🛡️ 1. The 200 SMA Macro Cash Shield
$$\text{Macro Condition} = \text{Close}_{\text{NIFTY}} > \text{SMA}_{200}(\text{NIFTY})$$
- **If TRUE**: The market is in an institutional bull phase. Evaluate sector momentum.
- **If FALSE**: The market is in a structural bear phase. Rotate **100% of capital into `GOLDBEES.NS` (Gold)**.

### 🔍 2. Absolute & Relative Momentum Ranking
1. **Absolute Momentum Gate**: Only sector ETFs trading above their own 100-day SMA qualify:
   $$\text{Price}_t > \text{SMA}_{100}(\text{Price})$$
2. **60-Day Relative Return**:
   $$\text{Return}_{60d} = \frac{\text{Price}_t - \text{Price}_{t-60}}{\text{Price}_{t-60}} \times 100$$
3. **Allocation**: Rank all eligible sectors and allocate 50% capital to each of the **Top 2 strongest sector ETFs**.

### ⏱️ 3. Execution & Tax Efficiency
- **Rebalance Frequency**: Every 10 trading days (bi-weekly).
- **Turnover**: Low (~10–12 trades/year).
- **Statutory Taxes**: Exact CNC delivery STT (0.10% buy + 0.10% sell), SEBI, GST 18%, Stamp Duty, and DP charges.

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/sector_rotation/run_test.py
```
