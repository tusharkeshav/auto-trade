# ─────────────────────────────────────────────────────────────────
#  run_wfo.py
#
#  Execute a Walk-Forward Optimization backtest.
# ─────────────────────────────────────────────────────────────────

from backtest.engines.walk_forward import WalkForwardOptimizer

# Configuration
SYMBOLS    = ["BTCUSDT", "ETHUSDT"]
INTERVAL   = "15m"
TRAIN_DAYS = 180
TEST_DAYS  = 60

if __name__ == "__main__":
    for sym in SYMBOLS:
        wfo = WalkForwardOptimizer(
            symbol=sym,
            interval=INTERVAL,
            train_days=TRAIN_DAYS,
            test_days=TEST_DAYS,
        )
        wfo.run()
