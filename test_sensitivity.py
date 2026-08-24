import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probability.signal_scorer import SignalScorer, TradeSignal
from backtest.engines.standard import BacktestEngine
from data.cache import DataCache

def is_nan(val):
    return val is None or pd.isna(val) or math.isnan(val)

class CustomScorer(SignalScorer):
    def __init__(self, symbol, max_adx, start_hour, end_hour):
        super().__init__(symbol=symbol, long_threshold=48.0, short_threshold=35.0)
        self.max_adx = max_adx
        self.start_hour = start_hour
        self.end_hour = end_hour
        # Set universal parameters
        self.long_threshold = 48.0
        
    def score(self, row: pd.Series) -> TradeSignal:
        price = row["close"]
        
        adx = row.get("adx", float('nan'))
        if is_nan(adx) or adx > self.max_adx:
            return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
            
        if hasattr(row, "name") and row.name is not None:
            hour = row.name.hour
            
            # Check time window
            if self.start_hour < self.end_hour:
                valid_hour = (self.start_hour <= hour < self.end_hour)
            else:
                valid_hour = (hour >= self.start_hour) or (hour < self.end_hour)
                
            if not valid_hour:
                return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
                
        # Call original logic but we need to bypass the macro filter that's currently in signal_scorer.py.
        # Actually, let's just temporarily patch signal_scorer.py or just copy the gating logic here.
        pass

import math

def run_sensitivity():
    print("Loading data...")
    cache = DataCache()
    df = cache.load("BTCUSDT", "15m")
    
    # Isolate the 2021-2023 period (first 90k candles out of the 180k downloaded)
    df = df.iloc[:90000]
    
    # We will just patch SignalScorer.score dynamically for each run.
    from probability import signal_scorer
    original_score = signal_scorer.SignalScorer.score
    
    adx_levels = [20.0, 25.0, 30.0]
    # (start, end). Note 16-24 means 16 to 23 inclusive.
    # 18-02 means 18 to 01 inclusive.
    # 14-22 means 14 to 21 inclusive.
    windows = [(14, 22), (16, 24), (18, 2)]
    
    results = []
    
    for max_adx in adx_levels:
        for start_h, end_h in windows:
            
            def patched_score(self, row: pd.Series) -> TradeSignal:
                price = row["close"]
                adx = row.get("adx", float('nan'))
                if is_nan(adx) or adx > max_adx:
                    return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
                
                if hasattr(row, "name") and row.name is not None:
                    hour = row.name.hour
                    if start_h < end_h:
                        valid = (start_h <= hour < end_h)
                    else:
                        valid = (hour >= start_h) or (hour < end_h)
                    
                    if not valid:
                        return TradeSignal(self.symbol, getattr(row, 'name', None), "NO_TRADE", 50.0, "LOW", 0.0, price, price, price, price, 0.0, [])
                
                # Now execute the standard gating
                # We can just call original_score BUT original_score in signal_scorer.py currently has a hardcoded MACRO filter!
                # We need to remove the hardcoded filter from signal_scorer.py first.
                
                return original_score(self, row)
            
            # Patch the method
            signal_scorer.SignalScorer.score = patched_score
            
            engine = BacktestEngine(symbol="BTCUSDT", interval="15m", candles=87000, threshold=48.0, atr_sl_mult=1.25)
            # Use cached df to avoid reloading
            engine.cache.load = lambda s, i: df
            
            res = engine.run()
            pf = res.profit_factor
            wr = res.win_rate
            trades = res.total_trades
            
            window_str = f"{start_h}-{end_h} UTC"
            print(f"ADX < {max_adx:<4} | {window_str:<10} => Trades: {trades:<4} | WR: {wr:>5.1f}% | PF: {pf:.2f}")
            results.append((max_adx, window_str, trades, wr, pf))
            
if __name__ == "__main__":
    run_sensitivity()
