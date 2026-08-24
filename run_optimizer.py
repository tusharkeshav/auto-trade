# ─────────────────────────────────────────────────────────────────
#  run_optimizer.py
#
#  Grid search runner for BTC parameter optimization.
#
#  Usage:
#      source .venv/bin/activate && python run_optimizer.py
#
#  Runtime: ~2-3 minutes (fetches 3 timeframes, runs 60 combos)
# ─────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.optimizer import GridOptimizer

if __name__ == "__main__":
    opt     = GridOptimizer(symbol="BTCUSDT")
    results = opt.run()
    opt.print_report(results)
