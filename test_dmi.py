"""
DMI Crossover Strategy — 4h BTC.
Entry: +DI crosses above -DI (LONG) or -DI crosses above +DI (SHORT)
       with ADX > 20 and ADX rising.
Exit: Opposite crossover, or trailing stop at 2x ATR.
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from data.cache import DataCache
from indicators import add_all_indicators
from config.settings import INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT

TAKER_FEE = 0.001

print("Loading 4h data...")
cache = DataCache()
df = cache.load('BTCUSDT', '4h')
df = add_all_indicators(df)
print(f"  {len(df)} candles ({df.index[0]} → {df.index[-1]})")

WARMUP = 300
capital = INITIAL_CAPITAL_USDT

# Test different configurations
configs = [
    ("DMI cross, 2xATR SL, 3R TP", False),
    ("DMI cross, 2xATR SL, trail at 1.5xATR", True),
]

for name, use_trail in configs:
    cash = capital
    trades = []
    i = WARMUP
    in_pos = False
    pos_dir = None

    while i < len(df) - 1:
        row = df.iloc[i]

        if not in_pos:
            # Check for DMI crossover
            di_plus = row.get("di_plus", float('nan'))
            di_minus = row.get("di_minus", float('nan'))
            adx = row.get("adx", float('nan'))
            adx_prev = df.iloc[i-1].get("adx", float('nan')) if i > 0 else float('nan')

            if math.isnan(di_plus) or math.isnan(di_minus) or math.isnan(adx):
                i += 1; continue

            # Need +DI/-DI from previous candle for crossover detection
            di_plus_prev = df.iloc[i-1].get("di_plus", float('nan')) if i > 0 else float('nan')
            di_minus_prev = df.iloc[i-1].get("di_minus", float('nan')) if i > 0 else float('nan')

            if math.isnan(di_plus_prev) or math.isnan(di_minus_prev):
                i += 1; continue

            # LONG: +DI crosses above -DI, ADX > 20 and rising
            if (di_plus_prev <= di_minus_prev and di_plus > di_minus
                    and adx > 20 and adx > adx_prev):
                entry = row["close"]
                atr = row.get("atr", float('nan'))
                if math.isnan(atr): atr = entry * 0.004
                sl_dist = atr * 2.0
                sl = entry - sl_dist
                tp = entry + sl_dist * 3.0
                size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
                if size <= 0: i += 1; continue
                in_pos = True; pos_dir = "LONG"; entry_idx = i

            # SHORT: -DI crosses above +DI, ADX > 20 and rising
            elif (di_minus_prev <= di_plus_prev and di_minus > di_plus
                    and adx > 20 and adx > adx_prev):
                entry = row["close"]
                atr = row.get("atr", float('nan'))
                if math.isnan(atr): atr = entry * 0.004
                sl_dist = atr * 2.0
                sl = entry + sl_dist
                tp = entry - sl_dist * 3.0
                size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
                if size <= 0: i += 1; continue
                in_pos = True; pos_dir = "SHORT"; entry_idx = i

            i += 1

        else:
            # Exit management
            row = df.iloc[i]
            ch, cl = row["high"], row["low"]
            pnl = None; ep = None

            if pos_dir == "LONG":
                # Check exit conditions
                # 1. Stop loss
                if cl <= sl:
                    pnl, ep = (sl - entry) * size, sl
                # 2. Trailing stop
                elif use_trail:
                    # Update trail: lock in gains as price moves up
                    new_sl = max(sl, ch - sl_dist * 1.5)
                    if new_sl > sl:
                        sl = new_sl
                    if cl <= sl:
                        pnl, ep = (sl - entry) * size, sl
                # 3. Take profit
                elif ch >= tp:
                    pnl, ep = (tp - entry) * size, tp
            else:
                if ch >= sl:
                    pnl, ep = (entry - sl) * size, sl
                elif use_trail:
                    new_sl = min(sl, cl + sl_dist * 1.5)
                    if new_sl < sl:
                        sl = new_sl
                    if ch >= sl:
                        pnl, ep = (entry - sl) * size, sl
                elif cl <= tp:
                    pnl, ep = (entry - tp) * size, tp

            if pnl is not None:
                fee = (entry + ep) * size * TAKER_FEE
                pnl -= fee
                cash += pnl
                trades.append(pnl)
                in_pos = False; pos_dir = None

            i += 1

    # Stats
    a = np.array(trades)
    wins = a[a > 0]; losses = a[a < 0]
    pf = round(sum(wins) / abs(sum(losses)), 2) if len(losses) else float('inf')
    wr = round(len(wins) / len(trades) * 100, 1) if trades else 0
    net = round(sum(a), 2)
    avg_w = round(np.mean(wins), 2) if len(wins) else 0
    avg_l = round(np.mean(losses), 2) if len(losses) else 0

    cap2, peak2, dd2 = capital, capital, 0.0
    for p in trades:
        cap2 += p
        if cap2 > peak2: peak2 = cap2
        dd = (peak2 - cap2) / peak2 * 100
        if dd > dd2: dd2 = dd

    print(f"\n{name}")
    print(f"  Trades: {len(trades)}")
    print(f"  WR: {wr}% ({len(wins)}W/{len(losses)}L)")
    print(f"  PF: {pf}")
    print(f"  Avg Win: ${avg_w} | Avg Loss: ${avg_l}")
    print(f"  Max DD: {round(dd2, 2)}%")
    print(f"  Net: ${net}")
    print(f"  Return: {round((cash/capital-1)*100, 2)}%")
