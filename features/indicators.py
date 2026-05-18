import numpy as np
import pandas as pd

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    df["return"] = prev_close.pct_change()
    df["log_return"] = np.log(prev_close / prev_close.shift(1))
    return df

def add_range(df: pd.DataFrame) -> pd.DataFrame:
    df["range"] = df["high"].shift(1) - df["low"].shift(1)
    return df

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    prev_volume = df["volume"].shift(1)
    df["ma10"] = prev_close.rolling(10).mean()
    df["ma50"] = prev_close.rolling(50).mean()
    df["ma10_ma50_diff"] = df["ma10"] - df["ma50"]
    df["vol_ma10"] = prev_volume.rolling(10).mean()
    df["price_ma10"] = prev_close / (df["ma10"] + 1e-10) - 1
    return df

def add_volatility(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df["volatility"] = df["close"].shift(1).rolling(window).std()
    return df

def add_rsi(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    if periods is None:
        periods = [7, 14]
    prev_close = df["close"].shift(1)
    for period in periods:
        delta = prev_close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    close_prev = df["close"].shift(2)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.rolling(window=period, min_periods=1).mean()
    return df

def add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ret = df["close"].shift(1).pct_change()
    df[f"rolling_vol_{window}"] = ret.rolling(window).std() * np.sqrt(252)
    return df

def add_ma_slope(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    if periods is None:
        periods = [10, 50]
    prev_close = df["close"].shift(1)
    for p in periods:
        ma_col = f"ma{p}"
        if ma_col not in df.columns:
            df[ma_col] = prev_close.rolling(p).mean()
        df[f"ma_slope_{p}"] = df[ma_col].pct_change(5)
    return df

def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    if "ma10" not in df.columns:
        df["ma10"] = prev_close.rolling(10).mean()
    if "ma50" not in df.columns:
        df["ma50"] = prev_close.rolling(50).mean()
    if "atr_14" not in df.columns:
        df = add_atr(df)
    df["trend_strength"] = (df["ma10"] - df["ma50"]).abs() / (df["atr_14"] + 1e-10)
    return df

def add_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    prev_vol = df["volume"].shift(1)
    vol_mean = prev_vol.rolling(window).mean()
    vol_std = prev_vol.rolling(window).std()
    df["volume_zscore"] = (prev_vol - vol_mean) / (vol_std + 1e-10)
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    ema_fast = prev_close.ewm(span=fast, adjust=False).mean()
    ema_slow = prev_close.ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    ma = prev_close.rolling(window).mean()
    std = prev_close.rolling(window).std()
    df["bollinger_upper"] = ma + num_std * std
    df["bollinger_lower"] = ma - num_std * std
    df["bollinger_pct"] = (prev_close - df["bollinger_lower"]) / (
        df["bollinger_upper"] - df["bollinger_lower"] + 1e-10
    )
    return df

def add_sma_crossover_signal(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    ma_fast = prev_close.rolling(fast).mean()
    ma_slow = prev_close.rolling(slow).mean()
    df["sma_cross_signal"] = (ma_fast > ma_slow).astype(int)
    return df