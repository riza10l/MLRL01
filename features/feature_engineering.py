"""
Feature Engineering Pipeline
=============================
Orchestrates all feature building from raw OHLCV data.
Applies basic + advanced indicators, target creation, and cleaning.

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd
from features.indicators import (
    add_returns, add_range, add_moving_averages, add_volatility,
    add_rsi, add_atr, add_rolling_volatility, add_ma_slope,
    add_trend_strength, add_volume_zscore, add_macd, add_bollinger_bands,
    add_sma_crossover_signal,
)


def build_all_features(df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
    """
    Build semua features dari raw OHLCV.
    
    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]
        include_advanced: Kalau True, tambahin RSI, ATR, MACD, dll (Prioritas 5)
    
    Returns:
        DataFrame with all features + target column
    """
    df = df.copy()
    
    # ── Basic Features (dari engine lama) ─────────────────────
    df = add_returns(df)
    df = add_range(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    
    # ── Advanced Features (Prioritas 5) ───────────────────────
    if include_advanced:
        df = add_rsi(df, periods=[7, 14])
        df = add_atr(df, period=14)
        df = add_rolling_volatility(df, window=20)
        df = add_ma_slope(df, periods=[10, 50])
        df = add_trend_strength(df)
        df = add_volume_zscore(df, window=20)
        df = add_macd(df)
        df = add_bollinger_bands(df)
        df = add_sma_crossover_signal(df)
    
    # ── Target: 1 = besok harga naik, 0 = turun/sama ─────────
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    # ── Clean NaN ─────────────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[FEAT] Total features: {len(get_feature_columns(df))} | Rows: {len(df):,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (exclude date, target, OHLCV, and non-numeric)."""
    exclude = {"date", "open", "high", "low", "close", "volume", "target",
               "log_return", "sma_cross_signal"}
    # Only include numeric columns to avoid string labels like 'bearish' causing errors in scaling
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_basic_feature_columns() -> list:
    """Feature columns dari engine lama."""
    return ["return", "range", "ma10", "ma50", "volatility",
            "ma10_ma50_diff", "vol_ma10", "price_ma10"]


def get_advanced_feature_columns() -> list:
    """All feature columns including Prioritas 5."""
    basic = get_basic_feature_columns()
    advanced = [
        "rsi_14", "rsi_7", "atr_14", "rolling_vol_20",
        "ma_slope_10", "ma_slope_50", "trend_strength", "volume_zscore",
        "macd", "macd_signal", "macd_hist",
        "bollinger_upper", "bollinger_lower", "bollinger_pct",
    ]
    return basic + advanced
