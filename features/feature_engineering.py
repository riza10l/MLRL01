
import numpy as np
import pandas as pd
from .indicators import (
    add_returns, add_range, add_moving_averages, add_volatility,
    add_rsi, add_atr, add_rolling_volatility, add_ma_slope,
    add_trend_strength, add_volume_zscore, add_macd, add_bollinger_bands,
    add_sma_crossover_signal,
)
from .features import build_production_features, robust_normalize, get_production_feature_columns

def build_all_features(
    df: pd.DataFrame,
    include_advanced: bool = True,
    horizon: int = 5,
    threshold: float = 0.005,
    normalize: bool = False,
) -> pd.DataFrame:

    df = df.copy()

    df = add_returns(df)
    df = add_range(df)
    df = add_moving_averages(df)
    df = add_volatility(df)

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

    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_ret > threshold).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feat_cols = get_feature_columns(df)

    if normalize:
        df = robust_normalize(df, feat_cols)

    print(f"[FEAT] Total features: {len(feat_cols)} | Rows: {len(df):,} | "
          f"horizon={horizon}d | target_mean={df['target'].mean():.2%}")
    return df


def build_features(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
    method: str = "triple_barrier",
    normalize: bool = False,
) -> pd.DataFrame:
    df = build_production_features(
        df,
        target_horizon=horizon,
        target_threshold=threshold,
        target_method=method,
    )

    if normalize:
        feat_cols = get_production_feature_columns(df)
        df = robust_normalize(df, feat_cols)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        "target", "target_return", "future_return",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_basic_feature_columns() -> list:
    return ["return", "range", "ma10", "ma50", "volatility",
            "ma10_ma50_diff", "vol_ma10", "price_ma10"]


def get_advanced_feature_columns(df: pd.DataFrame = None) -> list:
    if df is not None:
        return get_feature_columns(df)
    return get_basic_feature_columns() + [
        "log_return",
        "rsi_7", "rsi_14", "atr_14", "rolling_vol_20",
        "ma_slope_10", "ma_slope_50", "trend_strength", "volume_zscore",
        "macd", "macd_signal", "macd_hist",
        "bollinger_upper", "bollinger_lower", "bollinger_pct",
        "sma_cross_signal",
    ]