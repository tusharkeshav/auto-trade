import sys, os
import numpy as np
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probability.signal_scorer import SignalScorer, TradeSignal
from backtest.engines.standard import BacktestEngine
from data.cache import DataCache

# Ensure the scorer enforces our best macro rules
def is_nan(val):
    import math
    return val is None or pd.isna(val) or math.isnan(val)

def patched_score(self, row: pd.Series) -> TradeSignal:
    price = row["close"]
    adx = row.get("adx", float('nan'))
    if is_nan(adx) or adx > 25.0:
        return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
    
    if hasattr(row, "name") and row.name is not None:
        hour = row.name.hour
        day  = row.name.weekday()
        if hour < 16 or hour > 23 or day == 1:
            return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
    
    # Original scoring logic bypasses the hardcoded filter we removed earlier
    return original_score(self, row)

def run_monte_carlo(simulations=10000):
    print("Loading data and generating trades...")
    cache = DataCache()
    df = cache.load("BTCUSDT", "15m")
    
    # We will patch the scorer to ensure rules are perfect
    from probability import signal_scorer
    global original_score
    original_score = signal_scorer.SignalScorer.score
    signal_scorer.SignalScorer.score = patched_score
    
    # Run the engine
    engine = BacktestEngine(symbol="BTCUSDT", interval="15m", candles=180000, threshold=48.0, atr_sl_mult=1.25)
    engine.cache.load = lambda s, i: df
    res = engine.run()
    
    trades = res.trades
    if not trades:
        print("No trades generated.")
        return
        
    print(f"Generated {len(trades)} trades. Win Rate: {res.win_rate:.1f}%, PF: {res.profit_factor:.2f}")
    
    # Extract R-multiples (or just raw percentage returns)
    # The engine already uses fixed risk. We can extract the percentage P&L for each trade relative to the capital at that time,
    # but the simplest way is to extract the R-multiple.
    # We know winning trades make 1.5R and losing trades lose 1.0R (with some slippage/fees).
    # Let's extract the actual P&L divided by the risk_amount to get R.
    # Actually, simpler: extract the exact trade PnL as a percentage of the account at the start of the trade.
    
    trade_returns = []
    capital = engine.capital
    for t in trades:
        ret = t.pnl / capital
        trade_returns.append(ret)
        capital += t.pnl
        
    print(f"Running {simulations} Monte Carlo simulations...")
    
    max_drawdowns = []
    end_capitals = []
    
    for i in range(simulations):
        # Shuffle the sequence of returns
        shuffled = trade_returns.copy()
        random.shuffle(shuffled)
        
        sim_capital = 1.0  # start with 100%
        peak = 1.0
        max_dd = 0.0
        
        for ret in shuffled:
            sim_capital = sim_capital * (1 + ret)
            if sim_capital > peak:
                peak = sim_capital
            
            dd = (peak - sim_capital) / peak
            if dd > max_dd:
                max_dd = dd
                
        max_drawdowns.append(max_dd)
        end_capitals.append(sim_capital)
        
    max_drawdowns = np.array(max_drawdowns)
    
    print("\n=== MONTE CARLO DRAWDOWN ANALYSIS ===")
    print(f"Median Expected Drawdown: {np.percentile(max_drawdowns, 50) * 100:.2f}%")
    print(f"90th Percentile Drawdown: {np.percentile(max_drawdowns, 90) * 100:.2f}%")
    print(f"95th Percentile Drawdown: {np.percentile(max_drawdowns, 95) * 100:.2f}%")
    print(f"99th Percentile Drawdown: {np.percentile(max_drawdowns, 99) * 100:.2f}%")
    print(f"Absolute Worst Case DD:   {np.max(max_drawdowns) * 100:.2f}%")
    
if __name__ == "__main__":
    run_monte_carlo(10000)
