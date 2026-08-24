from dataclasses import dataclass

@dataclass
class CoinProfile:
    # --- Gate A: Location ---
    # ATR distance to S/R (in multipliers)
    sr_dist_near: float = 0.5
    sr_dist_mid:  float = 1.0
    sr_dist_far:  float = 1.5
    
    # Bollinger Band percentiles
    bb_lower_extreme: float = 0.1
    bb_lower_mid:     float = 0.3
    bb_upper_extreme: float = 0.9
    bb_upper_mid:     float = 0.7

    # --- Gate B: Confirmation ---
    # RSI
    rsi_oversold_extreme: float = 30.0
    rsi_oversold_mid:     float = 40.0
    rsi_overbought_extreme: float = 70.0
    rsi_overbought_mid:     float = 60.0

    # Stochastic RSI K
    stoch_oversold_extreme: float = 20.0
    stoch_oversold_mid:     float = 30.0
    stoch_overbought_extreme: float = 80.0
    stoch_overbought_mid:     float = 70.0

    # MACD Histogram (as a fraction of ATR)
    macd_near_zero: float = 0.1

    # --- Gate C: Context ---
    # Volume Ratio
    vol_ratio_surge: float = 1.5
    vol_ratio_high:  float = 1.0


# Default profile matches our highly tuned BTC parameters
DEFAULT_PROFILE = CoinProfile()

# Specific coin profiles (can be optimized later)
PROFILES = {
    "BTCUSDT": CoinProfile(
        # BTC matches default currently
    ),
    "ETHUSDT": CoinProfile(
        # ETH wicks deeper, might need different thresholds (to be optimized)
        rsi_oversold_extreme=25.0,
        rsi_overbought_extreme=75.0,
        stoch_oversold_extreme=15.0,
        stoch_overbought_extreme=85.0
    ),
}

def get_profile(symbol: str) -> CoinProfile:
    return PROFILES.get(symbol.upper(), DEFAULT_PROFILE)
