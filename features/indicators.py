"""
Features - Technical Indicators
================================
Semua technical indicators buat feature engineering.
Prioritas 5: RSI, ATR, rolling volatility, MA slope, trend strength, volume zscore, MACD, Bollinger.

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd


# ── Basic Indicators (dari trading_engine.py lama) ────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily log return & pct return."""
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_range(df: pd.DataFrame) -> pd.DataFrame:
    """High - Low range."""
    df["range"] = df["high"] - df["low"]
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """MA10, MA50, dan diff-nya."""
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma10_ma50_diff"] = df["ma10"] - df["ma50"]
    df["vol_ma10"] = df["volume"].rolling(10).mean()
    df["price_ma10"] = df["close"] / df["ma10"] - 1
    return df


def add_volatility(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling standard deviation of close."""
    df["volatility"] = df["close"].rolling(window).std()
    return df


# ── Prioritas 5: Richer State Representation ─────────────────

def add_rsi(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """Relative Strength Index."""
    if periods is None:
        periods = [7, 14]
    for period in periods:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.rolling(window=period).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling volatility (annualized std of returns)."""
    df[f"rolling_vol_{window}"] = df["return"].rolling(window).std() * np.sqrt(252)
    return df


def add_ma_slope(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """Slope of moving averages (pct change of MA)."""
    if periods is None:
        periods = [10, 50]
    for p in periods:
        ma_col = f"ma{p}"
        if ma_col not in df.columns:
            df[ma_col] = df["close"].rolling(p).mean()
        df[f"ma_slope_{p}"] = df[ma_col].pct_change(5)  # 5-day slope
    return df


def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Trend strength = |MA10 - MA50| / ATR."""
    if "ma10" not in df.columns:
        df["ma10"] = df["close"].rolling(10).mean()
    if "ma50" not in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()
    if "atr_14" not in df.columns:
        df = add_atr(df)
    df["trend_strength"] = (df["ma10"] - df["ma50"]).abs() / (df["atr_14"] + 1e-10)
    return df


def add_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume Z-score."""
    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, signal, histogram."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bollinger_upper"] = ma + num_std * std
    df["bollinger_lower"] = ma - num_std * std
    df["bollinger_pct"] = (df["close"] - df["bollinger_lower"]) / (
        df["bollinger_upper"] - df["bollinger_lower"] + 1e-10
    )
    return df


def add_sma_crossover_signal(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """SMA crossover signal for benchmark."""
    ma_fast = df["close"].rolling(fast).mean()
    ma_slow = df["close"].rolling(slow).mean()
    df["sma_cross_signal"] = (ma_fast > ma_slow).astype(int)
    return df
