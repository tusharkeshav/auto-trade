# ─────────────────────────────────────────────────────────────────
#  data/cache.py
#
#  Local OHLCV data cache — read/write to CSV (always available).
#  Parquet used automatically if pyarrow is installed (faster, smaller).
#
#  Usage:
#      from data.cache import DataCache
#      cache = DataCache()
#
#      # Save
#      cache.save(df, "BTCUSDT", "4h")
#
#      # Load (returns None if not cached)
#      df = cache.load("BTCUSDT", "4h")
#      if df is None:
#          df = fetch_from_api(...)
#          cache.save(df, "BTCUSDT", "4h")
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib  import Path
from typing   import Optional

import pandas as pd
from loguru import logger

# All cached files land here
CACHE_DIR = Path(__file__).parent.parent / "data" / "historical"

# Use parquet if available, else CSV
try:
    import pyarrow  # noqa: F401
    _FMT = "parquet"
except ImportError:
    _FMT = "csv"


class DataCache:
    """Thin wrapper around disk-based OHLCV storage."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str, candles: int = None) -> Path:
        ext = "parquet" if _FMT == "parquet" else "csv"
        if candles is not None:
            return self.cache_dir / f"{symbol}_{interval}_{candles}.{ext}"
        return self.cache_dir / f"{symbol}_{interval}.{ext}"

    # ── Public API ────────────────────────────────────────────────

    def _find_largest_cache_file(self, symbol: str, interval: str) -> Optional[Path]:
        """Finds the cached file with the largest candle count for the given symbol and interval."""
        ext = "parquet" if _FMT == "parquet" else "csv"
        best_path = None
        max_candles = -1

        for f in self.cache_dir.glob(f"{symbol}_{interval}*.{ext}"):
            # Expected formats: BTCUSDT_15m.csv or BTCUSDT_15m_87000.csv
            parts = f.stem.split("_")
            try:
                if len(parts) >= 3 and parts[-1].isdigit():
                    candles = int(parts[-1])
                else:
                    # Fallback for old cache format, assume small size
                    candles = 0 
                
                if candles > max_candles:
                    max_candles = candles
                    best_path = f
            except Exception:
                continue

        return best_path

    def exists(self, symbol: str, interval: str) -> bool:
        return self._find_largest_cache_file(symbol, interval) is not None

    def age_hours(self, symbol: str, interval: str) -> float:
        """How many hours since the cache file was last written."""
        p = self._find_largest_cache_file(symbol, interval)
        if not p:
            return float("inf")
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600

    def save(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        path = self._path(symbol, interval, len(df))
        if _FMT == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path)
        logger.success(f"Cached {len(df)} rows → {path.name}")
        return path

    def load(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        path = self._find_largest_cache_file(symbol, interval)
        if not path:
            return None
        if _FMT == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from cache ← {path.name}")
        return df

    def list_cached(self) -> list[dict]:
        """Return a summary of all cached files."""
        files = []
        for f in sorted(self.cache_dir.glob("*")):
            if f.suffix in (".parquet", ".csv"):
                parts = f.stem.split("_")   # e.g. BTCUSDT_4h
                symbol   = "_".join(parts[:-1])
                interval = parts[-1]
                age_h    = self.age_hours(symbol, interval)
                files.append({
                    "file":     f.name,
                    "symbol":   symbol,
                    "interval": interval,
                    "size_kb":  round(f.stat().st_size / 1024, 1),
                    "age_h":    round(age_h, 1),
                })
        return files
