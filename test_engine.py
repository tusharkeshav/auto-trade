import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Portfolio, OrderManager, Ledger
from dashboard import build_layout
from probability import SignalScorer

print("✅ All imports OK")

p = Portfolio()
l = Ledger()
s = SignalScorer()

print(f"✅ Portfolio  : cash=${p.cash:,.0f}")
print(f"✅ Ledger     : {len(l._records)} records")
print(f"✅ Scorer     : thresholds long={s.long_threshold}% / short={s.short_threshold}%")
print(f"✅ All systems ready — run main.py to start trading")
