# ─────────────────────────────────────────────────────────────────
#  config/india_settings.py
#  India market-specific configuration constants.
#  Mirrors config/settings.py structure. Import from here for India.
# ─────────────────────────────────────────────────────────────────

# ── India market symbols ────────────────────────────────────────
INDIA_SYMBOLS = ["NIFTY50", "BANKNIFTY"]
INDIA_DEFAULT_SYMBOL = "NIFTY50"

# ── Candle settings for India ───────────────────────────────────
# NSE intraday: 5m and 15m are most liquid.
# yfinance supports up to 60 days for 5m/15m intervals.
INDIA_DEFAULT_INTERVAL = "15m"
INDIA_DEFAULT_BARS     = 200      # 200 × 15m ≈ 2 NSE trading days

# ── Capital (INR paper account) ────────────────────────────────
INITIAL_CAPITAL_INR = 100_000.0       # ₹1 Lakh — retail benchmark account

# ── Risk management (India) ────────────────────────────────────
INDIA_MAX_RISK_PER_TRADE_PCT = 2.0   # 2.0% per trade — institutional swing sizing (~66% capital invested per trade)
INDIA_MAX_POSITION_PCT       = 30.0  # Indices are more liquid; allow 30%
INDIA_MAX_OPEN_TRADES        = 4     # Secular Leaders Alpha Book max open trades
INDIA_DAILY_LOSS_LIMIT_PCT   = 3.0   # Tighter than crypto — 3% daily limit

# ── Signal threshold ────────────────────────────────────────────
INDIA_SIGNAL_THRESHOLD = 50.0   # Marginally higher — NSE indices more efficient

# ── Exit parameters ───────────────────────────────────────────
INDIA_ATR_SL_MULT = 1.25       # Same as BTC: SL = 1.25 × ATR from support
INDIA_ATR_TP_MULT = 1.5        # TP = 1.5 × risk (1.5R)

# ── India VIX regime filter ───────────────────────────────────
# India VIX measures expected 30-day volatility of NIFTY 50.
# High VIX = risk-off / directional moves = mean-reversion fails.
# Low VIX = range-bound / institutional accumulation = mean-rev works.
#
# Empirical thresholds calibrated from NIFTY VIX history:
#   VIX < 12  → very low vol — mean-rev high probability
#   12–18     → normal range — standard mean-rev rules apply
#   18–25     → elevated vol — tighten filters (require stronger signal)
#   > 25      → stress regime — block all mean-rev, allow shorts only
VIX_LOW         = 12.0   # Below this: premium mean-rev zone
VIX_NORMAL_HIGH = 18.0   # Above this: elevated vol zone
VIX_STRESS      = 25.0   # Above this: stress — no long mean-rev trades

# ── NSE session macro shield ──────────────────────────────────
# NSE cash market: 09:15–15:30 IST.
# Best mean-reversion window: 09:30–11:30 IST and 13:00–14:30 IST.
# Avoid first 15 min (volatile open) and last 30 min (expiry games).
NSE_MACRO_START_HOUR_IST = 9    # IST hour: earliest entry
NSE_MACRO_START_MIN_IST  = 30   # IST minute: earliest entry
NSE_MACRO_END_HOUR_IST   = 15   # IST hour: last entry
NSE_MACRO_END_MIN_IST    = 0    # IST minute: last entry (15:00 IST)

# ── ADX macro shield ────────────────────────────────────────────
INDIA_MACRO_MAX_ADX = 25.0      # Same as BTC — block mean-rev above this

# ── Regime detection (India) ────────────────────────────────────
INDIA_REGIME_ADX_LOW         = 20.0
INDIA_REGIME_ADX_HIGH        = 30.0
INDIA_REGIME_BB_WIDTH_THRESH = 0.66

# ── Multi-Timeframe Dashboard UI Default Views (Trading Days) ──
# Configurable default visible range when switching intervals on web dashboard:
#   1d  -> 22 trading days (~1 month)
#   1h  -> 35 trading hours (~1 week = 7 hours × 5 days)
#   15m -> 75 trading bars  (~3 days = 25 bars × 3 days)
INDIA_UI_DEFAULT_VISIBLE_BARS = {
    "1d": 22,
    "1h": 35,
    "15m": 75,
}
