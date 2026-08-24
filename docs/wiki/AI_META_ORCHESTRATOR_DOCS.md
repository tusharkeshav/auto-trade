# 🤖 AI Quantitative Multi-Strategy Meta-Orchestrator Manual

## 1. System Architecture & Philosophy

The **AI Quantitative Multi-Strategy Meta-Orchestrator** is an autonomous master allocation engine. It replaces rigid single-strategy trading with an **adaptive multi-alpha portfolio manager**.

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │          AI / QUANT META-STRATEGY ORCHESTRATOR         │
                                  │                   (The "Master Brain")                 │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                    ┌─────────────────────────┴─────────────────────────┐
                                    ▼                                                   ▼
                     ┌─────────────────────────────┐                     ┌─────────────────────────────┐
                     │   LAYER 1: MARKET REGIME    │                     │  LAYER 2: STRATEGY SCORING  │
                     │      FEATURE VECTOR         │                     │     CONFIDENCE RANKING      │
                     ├─────────────────────────────┤                     ├─────────────────────────────┤
                     │ • NIFTY vs 200 SMA / 50 EMA │                     │ • Sector Rotation Conviction│
                     │ • ADX(14) Trend Strength    │                     │ • Large-Cap Pullback Score  │
                     │ • Volatility (ATR / Bands)  │                     │ • Minervini VCP Compression │
                     │ • Market Breadth (% > SMA50)│                     │ • SMC Order Block / FVG Zone│
                     │ • Sovereign Gold Momentum   │                     │ • Cash Shield Trigger       │
                     └─────────────────────────────┘                     └─────────────────────────────┘
                                    │                                                   │
                                    └─────────────────────────┬─────────────────────────┘
                                                              │
                                                              ▼
                                             ┌───────────────────────────────────┐
                                             │     LAYER 3: DYNAMIC CAPITAL      │
                                             │        OPTIMIZER / ROUTER         │
                                             └────────────────┬──────────────────┘
                                                              │
                 ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
                 ▼                                            ▼                                            ▼
   ┌───────────────────────────┐                ┌───────────────────────────┐                ┌───────────────────────────┐
   │    HIGH-MOMENTUM BULL     │                │   LOW-VOL CHOP / RANGE    │                │      CRISIS / DEFENSE     │
   ├───────────────────────────┤                ├───────────────────────────┤                ├───────────────────────────┤
   │ Allocates Capital To:     │                │ Allocates Capital To:     │                │ Allocates Capital To:     │
   │ • Sector ETF Momentum     │                │ • Large-Cap RS Pullbacks  │                │ • 100% GOLDBEES / Cash    │
   │ • Minervini VCP Breakouts │                │ • SMC Discount Setups     │                │   (Zero Equity Risk)      │
   │ • Blue-Chip Trend Riders  │                │ • Sovereign Gold Hedge    │                │                           │
   └───────────────────────────┘                └───────────────────────────┘                └───────────────────────────┘
```

---

## 2. Dynamic Weight Allocation Matrix

| Strategy / Asset Book | 🚀 Trending Bull (`ADX ≥ 22` & Bull) | 🔄 Choppy Sideways (`ADX < 22`) | 🛡️ Bear Defense (`NIFTY < 200 SMA`) |
|---|:---:|:---:|:---:|
| **Sector ETF Dual Momentum** | **45%** | 0% | 0% |
| **Minervini VCP Breakouts** | **30%** | 0% | 0% |
| **Large-Cap RS Pullbacks** | **25%** | **50%** | 0% |
| **Smart Money Concepts (SMC)** | 0% | **25%** | 0% |
| **Sovereign Gold (`GOLDBEES`)** | 0% | **25%** | **100%** |

---

## 3. Daily Execution Runbook (3:15 PM IST Routine)

1. Open your terminal at 3:15 PM IST (15 minutes before the NSE cash market close).
2. Execute the scanner command:
   ```bash
   source .venv/bin/activate
   python run_master_orchestrator.py --scan
   ```
3. Read the output table for today's active allocation and execute the recommended CNC Delivery orders in your broker.

---

## 4. Multi-Period Forensic Audit Command

To run the complete multi-year backtest and Monte Carlo verification:
```bash
python run_master_orchestrator.py --audit
```
