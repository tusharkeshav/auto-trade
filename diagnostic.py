import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.optimizers.grid import GridOptimizer
from rich.console import Console

console = Console()

def run_diagnostic():
    console.print("\n[bold magenta]🔍 Running Signal Probability Diagnostic...[/]")
    
    # Instantiate the optimizer to use its data loader and vectorized scorer
    opt = GridOptimizer(symbol="BTCUSDT", candles=87000)
    df = opt._load_data("15m")
    
    probs = df["_fast_prob"].dropna().values
    is_longs = df["_fast_is_long"].dropna().values
    
    long_probs = probs[is_longs]
    short_probs = probs[~is_longs]
    
    console.print(f"Total Candles: {len(df)}")
    console.print(f"LONG-biased candles: {len(long_probs)}")
    console.print(f"SHORT-biased candles: {len(short_probs)}\n")
    
    console.print("[bold cyan]LONG Probability Distribution:[/]")
    console.print(f"  Max Score:   {np.max(long_probs):.2f}")
    console.print(f"  99th %ile:   {np.percentile(long_probs, 99):.2f}")
    console.print(f"  95th %ile:   {np.percentile(long_probs, 95):.2f}")
    console.print(f"  90th %ile:   {np.percentile(long_probs, 90):.2f}")
    console.print(f"  75th %ile:   {np.percentile(long_probs, 75):.2f}")
    console.print(f"  Median:      {np.median(long_probs):.2f}\n")
    
    console.print("[bold yellow]Candle Counts by Threshold (LONG):[/]")
    thresholds = [45, 48, 50, 52, 55, 60, 65, 70, 72, 75]
    total_long = len(long_probs)
    for t in thresholds:
        count = np.sum(long_probs >= t)
        pct = (count / len(df)) * 100
        console.print(f"  ≥ {t}% : {count:>5} candles ({pct:.2f}% of all time)")

if __name__ == "__main__":
    run_diagnostic()
