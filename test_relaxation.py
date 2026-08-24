import sys, os
import pandas as pd
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probability.signal_scorer import SignalScorer, TradeSignal
from backtest.engines.standard import BacktestEngine
from data.cache import DataCache

def is_nan(val):
    return val is None or pd.isna(val) or math.isnan(val)

def run_relaxations():
    print("Loading 5-year BTC dataset (180,000 candles)...")
    cache = DataCache()
    df = cache.load("BTCUSDT", "15m")
    
    from probability import signal_scorer
    original_score = signal_scorer.SignalScorer.score
    
    # We will test several configurations:
    configs = [
        {"name": "Baseline (ADX<=25, 16-24 UTC, No Tue)", "adx": 25.0, "start_h": 16, "end_h": 24, "no_tue": True},
        {"name": "Relax Q1: Allow Tuesdays",             "adx": 25.0, "start_h": 16, "end_h": 24, "no_tue": False},
        {"name": "Relax Q2: Allow ADX<=30",              "adx": 30.0, "start_h": 16, "end_h": 24, "no_tue": False},
        {"name": "Relax Q3: Widen to 14-24 UTC",         "adx": 30.0, "start_h": 14, "end_h": 24, "no_tue": False},
        # Let's try one more pushing it to the absolute limit for N
        {"name": "Max Relax: ADX<=30, 14-02 UTC",        "adx": 30.0, "start_h": 14, "end_h": 2,  "no_tue": False},
    ]
    
    print("\nRunning backtests over 5 years. Target: Threshold=48.0, ATR_Mult=1.25\n")
    print(f"{'Config Name':<35} | {'Trades':<6} | {'WR %':<6} | {'PF':<5} | {'Max DD':<6} | {'P&L':<7}")
    print("-" * 80)
    
    for cfg in configs:
        def patched_score(self, row: pd.Series) -> TradeSignal:
            price = row["close"]
            adx = row.get("adx", float('nan'))
            if is_nan(adx) or adx > cfg["adx"]:
                return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
            
            if hasattr(row, "name") and row.name is not None:
                hour = row.name.hour
                day  = row.name.weekday()
                
                # Check Tuesday
                if cfg["no_tue"] and day == 1:
                    return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
                
                # Check Time
                sh, eh = cfg["start_h"], cfg["end_h"]
                if sh < eh:
                    valid = (sh <= hour < eh)
                else:
                    valid = (hour >= sh) or (hour < eh)
                    
                if not valid:
                    return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
            
            return original_score(self, row)
            
        signal_scorer.SignalScorer.score = patched_score
        
        # Engine configuration
        engine = BacktestEngine(symbol="BTCUSDT", interval="15m", candles=180000, threshold=48.0, atr_sl_mult=1.25)
        engine.cache.load = lambda s, i: df
        res = engine.run()
        
        pf = res.profit_factor
        wr = res.win_rate
        trades = res.total_trades
        dd = res.max_drawdown_pct
        pnl = res.total_pnl_pct
        
        print(f"{cfg['name']:<35} | {trades:<6} | {wr:>5.1f}% | {pf:.2f} | {dd:>5.1f}% | {pnl:>+6.1f}%")

if __name__ == "__main__":
    run_relaxations()
