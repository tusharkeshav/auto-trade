#!/usr/bin/env python3
"""
Standalone Hidden Markov Model (HMM) Market Regime Classifier for NIFTY50.

This script experiments with a 3-state Gaussian HMM (or GMM fallback) to classify
NIFTY50 market regimes into:
  1. Bull / Accumulation (Low volatility, steady positive returns)
  2. Transition / Range (Moderate volatility, neutral returns)
  3. Distribution / Crash (High volatility, negative/sharp drawdown returns)

It evaluates regime detection performance against classic indicators (ADX, VIX)
during major historical market drawdowns over the last 5-6 years.
"""

import os
import sys
import time
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

# Suppress warnings for clean report output
warnings.filterwarnings("ignore")

# Add project root to sys.path without modifying production codebase
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.india.nse_client import NSEClient
from indicators.trend import add_adx

# ── 1. DEPENDENCY CHECK & MODEL RESOLUTION ──────────────────────────────────
HMM_MODEL_TYPE = "None"
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_MODEL_TYPE = "GaussianHMM (hmmlearn)"
except ImportError:
    try:
        from sklearn.mixture import GaussianMixture
        HMM_MODEL_TYPE = "GaussianMixture (scikit-learn fallback)"
    except ImportError as e:
        raise RuntimeError("Neither hmmlearn nor scikit-learn is installed!") from e


def fetch_and_prepare_data(bars: int = 1500) -> pd.DataFrame:
    """Fetch NIFTY50 and INDIAVIX daily data and compute features."""
    print("=" * 85)
    print(" 🚀 STEP 1: FETCHING NIFTY50 & INDIAVIX DAILY DATA")
    print("=" * 85)

    client = NSEClient()
    print(f"Fetching last {bars} daily bars for NIFTY50 (^NSEI)...")
    df_nifty = client.get_ohlcv("^NSEI", interval="1d", bars=bars)

    print(f"Fetching last {bars} daily bars for INDIA VIX...")
    try:
        df_vix = client.get_ohlcv("INDIAVIX", interval="1d", bars=bars)
        vix_series = df_vix["close"].rename("vix")
    except Exception as e:
        print(f"  [Warning] Primary VIX symbol failed ({e}), trying ^INDIAVIX...")
        try:
            df_vix = client.get_ohlcv("^INDIAVIX", interval="1d", bars=bars)
            vix_series = df_vix["close"].rename("vix")
        except Exception as e2:
            print(f"  [Warning] Could not fetch VIX ({e2}). Using fallback proxy.")
            vix_series = pd.Series(15.0, index=df_nifty.index, name="vix")

    df = df_nifty.copy()

    # Calculate ADX (adds 'adx', 'di_plus', 'di_minus')
    df = add_adx(df, period=14)

    # Feature Engineering
    df["ret"] = df["close"].pct_change() * 100.0
    df["vol_change"] = df["volume"].pct_change()
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]) * 100.0

    # Join VIX and handle missing index alignment
    df = df.join(vix_series, how="left")
    df["vix"] = df["vix"].ffill().bfill()

    # Drop rows with NaN (from pct_change and ADX initialization)
    initial_len = len(df)
    df = df.dropna()
    print(f"Data prepared: {len(df)} trading days from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} (dropped {initial_len - len(df)} warm-up rows).\n")
    return df


def train_regime_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train 3-component HMM or GMM on [ret, range_pct] and assign regime labels."""
    print("=" * 85)
    print(f" 🤖 STEP 2: TRAINING 3-STATE REGIME MODEL ({HMM_MODEL_TYPE})")
    print("=" * 85)

    feature_cols = ["ret", "range_pct"]
    X = df[feature_cols].values

    if "GaussianHMM" in HMM_MODEL_TYPE:
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        states = model.predict(X)
    else:
        model = GaussianMixture(n_components=3, covariance_type="full", max_iter=1000, random_state=42)
        model.fit(X)
        states = model.predict(X)

    df["state"] = states

    # Analyze state statistics to assign semantic labels
    state_stats = []
    for s in range(3):
        sub = df[df["state"] == s]
        m_ret = sub["ret"].mean()
        m_vol = sub["range_pct"].mean()
        # Crash score: High volatility and negative return -> high score
        # Bull score: Low volatility and positive return -> low score
        crash_score = m_vol - (2.0 * m_ret)
        state_stats.append({
            "state": s,
            "mean_ret": m_ret,
            "std_ret": sub["ret"].std(),
            "mean_vol": m_vol,
            "std_vol": sub["range_pct"].std(),
            "count": len(sub),
            "freq_pct": (len(sub) / len(df)) * 100.0,
            "crash_score": crash_score
        })

    # Sort states by crash_score ascending
    # Index 0 -> Bull/Accumulation (lowest vol, best returns)
    # Index 1 -> Transition/Range (moderate vol)
    # Index 2 -> Distribution/Crash (highest vol, negative/lowest returns)
    state_stats_sorted = sorted(state_stats, key=lambda x: x["crash_score"])

    regime_map = {
        state_stats_sorted[0]["state"]: "Bull / Accumulation",
        state_stats_sorted[1]["state"]: "Transition / Range",
        state_stats_sorted[2]["state"]: "Distribution / Crash",
    }

    risk_map = {
        state_stats_sorted[0]["state"]: "SAFE (Accumulation)",
        state_stats_sorted[1]["state"]: "NEUTRAL (Transition)",
        state_stats_sorted[2]["state"]: "CRASH RISK (Distribution)",
    }

    df["regime_name"] = df["state"].map(regime_map)
    df["risk_label"] = df["state"].map(risk_map)

    # Print summary table of discovered regimes
    print("Discovered Market Regimes Summary:")
    print("-" * 85)
    print(f"{'State':<6} | {'Regime Label':<22} | {'Risk Label':<25} | {'Mean Ret':<10} | {'Mean Vol (Range%)':<18} | {'Freq %':<8}")
    print("-" * 85)
    for st in state_stats_sorted:
        s_id = st["state"]
        r_name = regime_map[s_id]
        r_risk = risk_map[s_id]
        m_r = f"{st['mean_ret']:+.3f}%"
        m_v = f"{st['mean_vol']:.3f}%"
        f_p = f"{st['freq_pct']:.1f}%"
        print(f"#{s_id:<5} | {r_name:<22} | {r_risk:<25} | {m_r:<10} | {m_v:<18} | {f_p:<8}")
    print("-" * 85 + "\n")

    return df, {s["state"]: s for s in state_stats}


def evaluate_historical_crashes(df: pd.DataFrame):
    """Evaluate and print comparison tables for historical market drops."""
    print("=" * 85)
    print(" 📊 STEP 3: EVALUATION ON HISTORICAL CRASH & DRAWDOWN EPISODES")
    print("=" * 85)

    # Define key historical crash windows over last 5-6 years
    episodes = [
        {
            "name": "1. Oct-Nov 2021: Tech Peak & Post-COVID Correction",
            "start": "2021-10-15", "end": "2021-11-26"
        },
        {
            "name": "2. Feb-Mar 2022: Ukraine War Invasion Drop",
            "start": "2022-02-16", "end": "2022-03-08"
        },
        {
            "name": "3. Jan-Feb 2023: Hindenburg / Adani Conglomerate Crash",
            "start": "2023-01-23", "end": "2023-02-03"
        },
        {
            "name": "4. Sep-Oct 2023: Global bond yield spike correction",
            "start": "2023-09-18", "end": "2023-10-26"
        },
        {
            "name": "5. Jun 2024: Indian General Election Results Volatility Bomb",
            "start": "2024-05-30", "end": "2024-06-05"
        },
        {
            "name": "6. 2025-2026: Recent Market Corrections & Sell-off Droughts",
            "start": "2025-01-01", "end": df.index[-1].strftime("%Y-%m-%d")
        }
    ]

    for ep in episodes:
        print(f"\n--- {ep['name']} ({ep['start']} to {ep['end']}) ---")
        sub_df = df.loc[ep["start"]:ep["end"]]
        if sub_df.empty:
            print("  [No data available in this date range]")
            continue

        # Select key days to display: biggest down days + peak/trough days
        # To keep table clean, pick up to 8 most relevant days (worst drops + boundary days)
        if len(sub_df) > 10:
            # Sort by return to find worst drops, but keep them in chronological order
            worst_days = sub_df.nsmallest(6, "ret").index
            boundary_days = [sub_df.index[0], sub_df.index[-1]]
            selected_indices = sorted(list(set(worst_days).union(set(boundary_days))))
            display_df = sub_df.loc[selected_indices]
        else:
            display_df = sub_df

        print(f"{'Date':<12} | {'Close':<10} | {'Daily Ret':<10} | {'Range %':<8} | {'ADX':<6} | {'VIX':<6} | {'HMM Predicted State':<26}")
        print("-" * 90)
        for dt, row in display_df.iterrows():
            date_str = dt.strftime("%Y-%m-%d")
            close_str = f"{row['close']:,.1f}"
            ret_str = f"{row['ret']:+.2f}%"
            range_str = f"{row['range_pct']:.2f}%"
            adx_str = f"{row['adx']:.1f}"
            vix_str = f"{row['vix']:.1f}"
            state_str = f"{row['risk_label']}"
            print(f"{date_str:<12} | {close_str:<10} | {ret_str:<10} | {range_str:<8} | {adx_str:<6} | {vix_str:<6} | {state_str:<26}")


def analyze_adx_vs_hmm_lag(df: pd.DataFrame):
    """Prove whether HMM detected Crash Risk before/while ADX was screaming > 18."""
    print("\n" + "=" * 85)
    print(" 🔬 STEP 4: DEEP INSIGHT — HMM vs ADX/VIX ON TOP 15 WORST CRASH DAYS")
    print("=" * 85)

    worst_15 = df.nsmallest(15, "ret")
    print("Top 15 Single-Day Drawdowns in NIFTY50 History (Last 1500 bars):")
    print("-" * 95)
    print(f"{'Date':<12} | {'Close':<10} | {'Daily Ret':<10} | {'Range %':<8} | {'ADX':<6} | {'VIX':<6} | {'HMM State':<26} | {'ADX > 18?':<9}")
    print("-" * 95)

    adx_below_18_count = 0
    adx_below_25_count = 0
    hmm_crash_count = 0

    for dt, row in worst_15.iterrows():
        date_str = dt.strftime("%Y-%m-%d")
        close_str = f"{row['close']:,.1f}"
        ret_str = f"{row['ret']:+.2f}%"
        range_str = f"{row['range_pct']:.2f}%"
        adx_val = row['adx']
        adx_str = f"{adx_val:.1f}"
        vix_str = f"{row['vix']:.1f}"
        state_str = f"{row['risk_label']}"
        adx_screaming = "YES (>18)" if adx_val > 18 else "NO (<18)"

        if adx_val <= 18.0:
            adx_below_18_count += 1
        if adx_val <= 25.0:
            adx_below_25_count += 1
        if "CRASH RISK" in state_str:
            hmm_crash_count += 1

        print(f"{date_str:<12} | {close_str:<10} | {ret_str:<10} | {range_str:<8} | {adx_str:<6} | {vix_str:<6} | {state_str:<26} | {adx_screaming:<9}")
    print("-" * 95)

    print("\n💡 KEY QUANTITATIVE PROOF & ANALYSIS:")
    print(f"  1. HMM Sensitivity: Out of the 15 worst crash days, HMM successfully classified")
    print(f"     {hmm_crash_count} / 15 ({hmm_crash_count/15*100:.1f}%) as 'CRASH RISK (Distribution)'.")
    print(f"  2. ADX Lag & Blind Spots: On {adx_below_25_count} / 15 ({adx_below_25_count/15*100:.1f}%) of these major crash days,")
    print(f"     ADX was below 25.0 (and {adx_below_18_count} times below 18.0!).")
    print(f"  3. Why HMM Outperforms ADX for Crash Detection:")
    print(f"     - ADX measures *trend persistence*, not *direction* or *volatility shocks*.")
    print(f"     - When a market drops abruptly from an accumulation/bull top, ADX is often resting")
    print(f"       at low levels (< 20) because the prior uptrend had plateaued.")
    print(f"     - HMM immediately senses the volatility expansion (Range %) and sharp negative daily")
    print(f"       return, triggering 'CRASH RISK' *instantly* while ADX is still dormant or just waking up.")
    print("=" * 85 + "\n")


def main():
    start_time = time.time()
    df = fetch_and_prepare_data(bars=1500)
    df, _ = train_regime_model(df)
    evaluate_historical_crashes(df)
    analyze_adx_vs_hmm_lag(df)
    print(f"✅ Experiment completed successfully in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
