import re
import pandas as pd
from datetime import timedelta
from backtest.engines.walk_forward import WalkForwardOptimizer

def run():
    wfo = WalkForwardOptimizer(symbol="BTCUSDT", interval="15m", train_days=90, test_days=30)
    df = wfo._load_data()
    
    # We need to get the trades
    # Let's run a short WFO or just load the results...
    pass
