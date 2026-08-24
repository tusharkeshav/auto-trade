# 🏛️ Strategy 4: All-Weather 50/50 Dual-Book Engine

## 1. Executive Overview
- **Philosophy**: Combines two uncorrelated quantitative engines (50% Sector ETF Dual-Momentum + 50% Large-Cap Pullback with 60d RS Filter) into a synchronized portfolio.
- **Why it Works**: When sector momentum pauses during choppy markets, the pullback engine generates steady income from dip-rebounds. When markets crash, the 200 SMA shield rotates capital into Gold, cutting standalone drawdown from -28.7% down to -16.0%.

---

## 2. Allocation & Engine Breakdown

```
Total Capital: ₹1,00,000 Base
├── Book 1 (50% = ₹50,000): Sector ETF Dual-Momentum (ITBEES, BANKBEES, AUTOBEES, CPSEETF)
└── Book 2 (50% = ₹50,000): Large-Cap Pullback Engine (8 Blue-Chips + 60d RS Gate)
```

### 🎯 Key Performance Targets
- **Target CAGR**: $>12.0\%$ per year.
- **Achieved 5-Year CAGR**: **13.50%** (1.75× NIFTY 50 benchmark).
- **Max Drawdown**: **-16.04%** (Halved compared to standalone sector ETFs).
- **Sortino Ratio**: **1.92** (Institutional downside quality).

---

## 3. How to Run

```bash
source .venv/bin/activate
python strategies/all_weather_dual_book/run_test.py
```
