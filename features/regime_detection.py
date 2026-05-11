"""
Regime Detection
=================
Prioritas 6: Regime Awareness (bull/bear, high_vol/low_vol)
Prioritas 10: HMM Regime Detection (Hidden Markov Model)

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd


# ── Prioritas 6: Simple Regime Detection ──────────────────────

def detect_trend_regime(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Deteksi regime berdasarkan trend:
    - bullish: close > MA(lookback)
    - bearish: close < MA(lookback)
    """
    df = df.copy()
    ma = df["close"].rolling(lookback).mean()
    df["trend_regime"] = np.where(df["close"] > ma, 1, 0)  # 1=bullish, 0=bearish
    df["trend_regime_label"] = np.where(df["trend_regime"] == 1, "bullish", "bearish")
    return df


def detect_volatility_regime(df: pd.DataFrame, lookback: int = 20, threshold_quantile: float = 0.7) -> pd.DataFrame:
    """
    Deteksi regime berdasarkan volatility.
    """
    df = df.copy()
    # Fill NaN returns with 0 to avoid cascade NaNs in rolling std
    returns = df["close"].pct_change().fillna(0)
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
    
    # Handle case where all rolling_vol are NaN
    if rolling_vol.isna().all():
        df["vol_regime"] = 0
        df["vol_regime_label"] = "low_vol"
        df["regime_vol_value"] = 0.0
        return df

    threshold = rolling_vol.quantile(threshold_quantile)
    # Ensure threshold is not NaN
    if np.isnan(threshold):
        threshold = rolling_vol.mean()
    
    df["vol_regime"] = np.where(rolling_vol > threshold, 1, 0)
    df["vol_regime_label"] = np.where(df["vol_regime"] == 1, "high_vol", "low_vol")
    df["regime_vol_value"] = rolling_vol.fillna(0)
    return df


def detect_combined_regime(df: pd.DataFrame, trend_lookback: int = 50, vol_lookback: int = 20) -> pd.DataFrame:
    """
    Combined regime: sideways, steady_up, crash, volatile_rally.
    """
    df = detect_trend_regime(df, trend_lookback)
    df = detect_volatility_regime(df, vol_lookback)
    df["combined_regime"] = df["trend_regime"] + df["vol_regime"] * 2
    regime_names = {0: "sideways", 1: "steady_up", 2: "crash", 3: "volatile_rally"}
    df["combined_regime_label"] = df["combined_regime"].map(regime_names)
    # Add a fallback hmm_regime column for consistency
    df["hmm_regime"] = df["combined_regime"]
    return df


# ── Prioritas 10: Statistical Regime Detection (HMM Alternative) ───────────────────────

def detect_statistical_regime(df: pd.DataFrame, n_states: int = 3, lookback: int = 50) -> pd.DataFrame:
    """
    Statistical regime detection tanpa HMM - pake clustering sederhana.
    Lebih robust dan ga butuh compiler.
    """
    df = df.copy()
    
    # Ensure we have return and volatility features
    if "return" not in df.columns:
        df["return"] = df["close"].pct_change().fillna(0)
    if "volatility" not in df.columns:
        returns = df["close"].pct_change().fillna(0)
        df["volatility"] = returns.rolling(20).std() * np.sqrt(252)
    
    # Clean data
    features_df = df[["return", "volatility"]].fillna(0)
    
    # Normalize features
    returns_norm = (features_df["return"] - features_df["return"].mean()) / (features_df["return"].std() + 1e-8)
    vol_norm = (features_df["volatility"] - features_df["volatility"].mean()) / (features_df["volatility"].std() + 1e-8)
    
    # Simple quantile-based regime classification
    if n_states == 3:
        # 3 states: bearish (0), neutral (1), bullish (2)
        return_q33 = returns_norm.quantile(0.33)
        return_q67 = returns_norm.quantile(0.67)
        
        regime = np.zeros(len(df), dtype=int)
        regime[returns_norm > return_q67] = 2  # bullish
        regime[(returns_norm > return_q33) & (returns_norm <= return_q67)] = 1  # neutral
        # bearish stays 0
        
        df["hmm_regime"] = regime
        
    elif n_states == 4:
        # 4 states: crash (0), bearish (1), bullish (2), volatile_rally (3)
        return_median = returns_norm.median()
        vol_median = vol_norm.median()
        
        regime = np.zeros(len(df), dtype=int)
        regime[(returns_norm > return_median) & (vol_norm <= vol_median)] = 2  # bullish steady
        regime[(returns_norm > return_median) & (vol_norm > vol_median)] = 3   # volatile rally
        regime[(returns_norm <= return_median) & (vol_norm > vol_median)] = 0  # crash
        regime[(returns_norm <= return_median) & (vol_norm <= vol_median)] = 1 # bearish steady
        
        df["hmm_regime"] = regime
        
    else:
        # Default to combined regime
        print(f"[REGIME] Statistical regime supports 3-4 states. Using combined regime.")
        return detect_combined_regime(df)
    
    # Smooth regime dengan rolling mode
    df["hmm_regime"] = df["hmm_regime"].rolling(window=5, min_periods=1).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
    ).astype(int)
    
    # Calculate state statistics
    state_returns = {}
    for state in range(n_states):
        mask = df["hmm_regime"] == state
        if mask.any():
            state_returns[state] = df.loc[mask, "return"].mean()
        else:
            state_returns[state] = 0.0
    
    print(f"[REGIME] Statistical regime fitted with {n_states} states.")
    for s, r in sorted(state_returns.items()):
        print(f"  State {s}: mean_return = {r:.6f}")
    
    # Also add simple regimes for completeness
    df = detect_combined_regime(df)
    return df


def add_regime_features(df: pd.DataFrame, method: str = "volatility_trend",
                        n_hmm_states: int = 3) -> pd.DataFrame:
    """
    Main entry point: add regime features ke dataframe.
    
    Args:
        method: "volatility_trend" atau "statistical" (replaces "hmm")
        n_hmm_states: jumlah states buat statistical regime (3 atau 4)
    """
    if method == "hmm" or method == "statistical":
        # Use statistical regime (no compiler needed)
        df = detect_statistical_regime(df, n_states=n_hmm_states)
    else:
        df = detect_combined_regime(df)
    
    return df