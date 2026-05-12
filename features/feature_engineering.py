"""
Feature Engineering Pipeline (V4 — Leakage-Free)
==================================================
Thin wrapper around features.py for backward compatibility.
All actual feature logic is in features.py.

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd
from .indicators import (
    add_returns, add_range, add_moving_averages, add_volatility,
    add_rsi, add_atr, add_rolling_volatility, add_ma_slope,
    add_trend_strength, add_volume_zscore, add_macd, add_bollinger_bands,
    add_sma_crossover_signal,
)


def build_all_features(df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
    """
    Build features from raw OHLCV using indicator functions.
    ALL indicators now use shift(1) internally — no lookahead.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]
        include_advanced: If True, add RSI, ATR, MACD, etc.

    Returns:
        DataFrame with all features + target column
    """
    df = df.copy()

    # ── Basic Features ────────────────────────────────────────
    df = add_returns(df)
    df = add_range(df)
    df = add_moving_averages(df)
    df = add_volatility(df)

    # ── Advanced Features ─────────────────────────────────────
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

    # ── Target: multi-day forward return (NO shift(-1) leakage) ──
    # Use 5-day forward return with 0.5% threshold
    horizon = 5
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_ret > 0.005).astype(int)

    # ── Clean NaN ─────────────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[FEAT] Total features: {len(get_feature_columns(df))} | Rows: {len(df):,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (exclude date, target, OHLCV, non-numeric)."""
    exclude = {"date", "open", "high", "low", "close", "volume", "target",
               "target_return", "future_return", "log_return", "sma_cross_signal"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_basic_feature_columns() -> list:
    """Feature columns from basic indicators."""
    return ["return", "range", "ma10", "ma50", "volatility",
            "ma10_ma50_diff", "vol_ma10", "price_ma10"]


def get_advanced_feature_columns() -> list:
    """All feature columns including advanced."""
    basic = get_basic_feature_columns()
    advanced = [
        "rsi_14", "rsi_7", "atr_14", "rolling_vol_20",
        "ma_slope_10", "ma_slope_50", "trend_strength", "volume_zscore",
        "macd", "macd_signal", "macd_hist",
        "bollinger_upper", "bollinger_lower", "bollinger_pct",
    ]
    return basic + advanced
