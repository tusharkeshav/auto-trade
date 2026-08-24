# ─────────────────────────────────────────────────────────────────
#  config/settings.py  —  Global bot configuration
# ─────────────────────────────────────────────────────────────────

# ── Binance REST API ──────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# ── Live trading symbols ─────────────────────
# SOL excluded — proved structurally incompatible (retail-driven, ignores liquidity clock)
SYMBOLS        = ["BTCUSDT", "ETHUSDT"]
DEFAULT_SYMBOL = "BTCUSDT"

# ── Primary candle settings (15m — all symbols) ─────────────────
# Validated exclusively on 15m candles. Do NOT change to 1h.
DEFAULT_INTERVAL = "15m"
DEFAULT_LIMIT    = 200          # 200 × 15m ≈ 2 days — sufficient for all indicator warmup

# ── Secondary timeframe (30m — BTC only) ─────────────────────────
# Research confirmed: 30m edge exists on BTC (PF 1.19, DD 8.8%).
# ETH excluded — 15m edge on ETH needs re-validation first.
# Overlap with 15m is only 9% → signals are genuinely independent.
SECONDARY_INTERVAL = "30m"
SECONDARY_LIMIT    = 200        # 200 × 30m ≈ 4 days — sufficient for indicator warmup
SECONDARY_SYMBOLS  = ["BTCUSDT"]  # ETH added here only after its 15m PF is re-validated

# ── Paper trading capital ─────────────────────
INITIAL_CAPITAL_USDT = 10_000.0

# ── Risk management ───────────────────────────
# Kelly Criterion position sizing.
# f* = W - (1-W) / (avgWin/avgLoss)
# Uses historical OOS WFO data: 53% WR, avg win $342, avg loss $234
BACKTEST_WIN_RATE            = 0.53
BACKTEST_AVG_WIN_USDT        = 342.30
BACKTEST_AVG_LOSS_USDT       = 234.35
KELLY_FRACTION               = BACKTEST_WIN_RATE - (1 - BACKTEST_WIN_RATE) / (BACKTEST_AVG_WIN_USDT / BACKTEST_AVG_LOSS_USDT)
# Kelly = 0.38 → quarter-Kelly = 0.095 → 9.5% → capped at 2%
MAX_RISK_PER_TRADE_PCT = min(2.0, round(KELLY_FRACTION * 0.25 * 100, 1))
MAX_POSITION_PCT       = 20.0   # Allow up to 20% of cash in one position (for BTC sizing)
MAX_OPEN_TRADES        = 2      # One position per symbol — max 2 simultaneous (BTC + ETH)
DAILY_LOSS_LIMIT_PCT   = 5.0

# ── Strategy: Entry threshold ─────────────────
# Minimum probability score to open a trade. Validated at 48.0.
# Lower = more trades, lower PF. Higher = fewer trades, higher PF.
SIGNAL_THRESHOLD = 48.0

# ── Strategy: Exit parameters ─────────────────
# All-In, All-Out: single exit at 1.5R take profit, 1.0R stop loss.
ATR_SL_MULT   = 1.25            # Stop-loss = 1.25 × ATR from support
ATR_TP_MULT   = 1.5             # Take-profit = 1.5 × risk from entry

# ── Macro Shield ────────────────────────────────────────
# Structural edge: mean reversion works during institutional liquidity exhaustion.
# ADX > 25 means the market is trending — mean reversion fails in trends.
# Outside 16-24 UTC, institutional directional flow dominates — avoid.
MACRO_MAX_ADX        = 25.0     # Block mean-rev trades when ADX > 25 (tested threshold)
MACRO_SESSION_START  = 16       # UTC hour: session open
MACRO_SESSION_END    = 24       # UTC hour: session close (24 = midnight)

# ── Regime Detection ────────────────────────────────────
# Switches between mean-reversion and momentum based on ADX + BB width.
REGIME_ADX_LOW         = 20.0   # ADX ≤ this → strong mean-reversion zone
REGIME_ADX_HIGH        = 30.0   # ADX ≥ this → strong momentum zone
REGIME_BB_WIDTH_THRESH = 0.66   # BB width percentile threshold in hybrid zone

# ── Momentum Strategy ───────────────────────────────────
# Uses 4h timeframe for trend-following (validated: PF 1.65).
# Fires when ADX ≥ MOMENTUM_ADX_MIN and structures align.
MOMENTUM_THRESHOLD      = 60.0   # Minimum probability for momentum entry
MOMENTUM_ADX_MIN        = 20.0   # ADX must be ≥ this for momentum to fire
MOMENTUM_ATR_SL_MULT    = 2.0    # Stop-loss = 2.0 × ATR (wider for trends)
MOMENTUM_ATR_TP_RATIO   = 3.0    # Take-profit = 3.0 × risk (let trends run)
MOMENTUM_INTERVAL       = "4h"   # Candle timeframe
MOMENTUM_LIMIT          = 200    # 200 × 4h ≈ 33 days for indicator warmup
MOMENTUM_SYMBOLS        = ["BTCUSDT"]  # BTC only — ETH 4h not tested yet

# ── Web dashboard ─────────────────────────────
WEB_DASHBOARD_PORT     = 8080
