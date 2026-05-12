"""
Features V4 — Leakage-Free Professional Feature Engineering
=============================================================
CRITICAL RULES:
  1. ALL features use shift(1) — only past data available at decision time
  2. NO future information via shift(-N)
  3. Target uses explicit forward-looking window with proper isolation
  4. feature columns exclude ALL target-related and raw price columns

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════
#  TARGET ENGINEERING — Triple Barrier Labeling
# ═══════════════════════════════════════════════════════════════

def create_target(df: pd.DataFrame, horizon: int = 5,
                  threshold: float = 0.005, method: str = "triple_barrier") -> pd.DataFrame:
    """
    Professional target labeling. All methods produce forward-looking labels
    that are STRICTLY isolated from features.

    Methods:
      - triple_barrier: ATR-adjusted barriers (institutional standard)
      - threshold: fixed percentage threshold
      - binary: simple up/down

    Returns df with 'target' column (0 or 1) and 'target_return' (continuous).
    """
    df = df.copy()

    if method == "triple_barrier":
        df = _triple_barrier_target(df, horizon=horizon, atr_mult=1.5)
    elif method == "threshold":
        # Forward return over horizon
        future_ret = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (future_ret > threshold).astype(int)
        df['target_return'] = future_ret
    elif method == "binary":
        future_ret = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (future_ret > 0).astype(int)
        df['target_return'] = future_ret
    else:
        future_ret = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (future_ret > threshold).astype(int)
        df['target_return'] = future_ret

    return df


def _triple_barrier_target(df: pd.DataFrame, horizon: int = 5,
                           atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Triple Barrier Labeling (de Prado, 2018).
    - Upper barrier: +atr_mult * ATR (take profit)
    - Lower barrier: -atr_mult * ATR (stop loss)
    - Vertical barrier: horizon bars (time limit)

    Label: 1 if upper barrier hit first, 0 otherwise.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Compute ATR for barrier sizing (using PAST data only, shift(1))
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        )
    )
    tr[0] = highs[0] - lows[0]  # First bar: no previous close
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    n = len(df)
    labels = np.full(n, np.nan)
    target_returns = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = closes[i]
        barrier_atr = atr[i] * atr_mult

        upper = entry_price + barrier_atr
        lower = entry_price - barrier_atr

        # Scan forward bars
        hit_upper = False
        hit_lower = False
        exit_price = closes[min(i + horizon, n - 1)]

        for j in range(i + 1, min(i + horizon + 1, n)):
            if highs[j] >= upper:
                hit_upper = True
                exit_price = upper
                break
            if lows[j] <= lower:
                hit_lower = True
                exit_price = lower
                break

        if hit_upper:
            labels[i] = 1
        elif hit_lower:
            labels[i] = 0
        else:
            # Vertical barrier — use direction of final return
            final_ret = (exit_price - entry_price) / (entry_price + 1e-10)
            labels[i] = 1 if final_ret > 0 else 0

        target_returns[i] = (exit_price - entry_price) / (entry_price + 1e-10)

    df['target'] = labels
    df['target_return'] = target_returns

    return df


# ═══════════════════════════════════════════════════════════════
#  CORE FEATURE PIPELINE — Leakage-Free
# ═══════════════════════════════════════════════════════════════

def build_production_features(df: pd.DataFrame,
                               target_horizon: int = 5,
                               target_threshold: float = 0.005,
                               target_method: str = "triple_barrier") -> pd.DataFrame:
    """
    V4 Leakage-Free Feature Engineering Pipeline.

    RULES:
      - ALL features use shift(1) where they reference current-bar OHLCV
      - This ensures features only contain information available BEFORE
        the trading decision is made (at market open)
      - Target is computed separately and never mixed into features
    """
    df = df.copy()

    # Ensure date column exists and is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # ── STEP 1: Compute raw intermediates (no shift yet) ──────
    # These are helper columns; final features will use shift(1)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Previous-bar values for feature computation
    close_prev = close.shift(1)
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    volume_prev = volume.shift(1)

    # ── STEP 2: ATR (using shift(1) — only past data) ────────
    tr = pd.concat([
        high_prev - low_prev,
        (high_prev - close.shift(2)).abs(),
        (low_prev - close.shift(2)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # ── STEP 3: Returns (using shift(1)) ─────────────────────
    df['ret_1d'] = close_prev.pct_change(1)
    df['ret_5d'] = close_prev.pct_change(5)
    df['ret_20d'] = close_prev.pct_change(20)

    # ── STEP 4: Trend Features ───────────────────────────────
    ema20 = close_prev.ewm(span=20, adjust=False).mean()
    ema50 = close_prev.ewm(span=50, adjust=False).mean()
    df['close_vs_ema20'] = (close_prev - ema20) / (df['atr_14'] + 1e-8)
    df['close_vs_ema50'] = (close_prev - ema50) / (df['atr_14'] + 1e-8)
    df['ema_slope_20'] = ema20.pct_change(5)
    df['ema_slope_50'] = ema50.pct_change(10)
    df['trend_strength'] = (ema20 - ema50).abs() / (df['atr_14'] + 1e-8)

    # ── STEP 5: ADX (using shift(1) data) ────────────────────
    df = _add_adx(df, period=14)

    # ── STEP 6: Trend Persistence ────────────────────────────
    ret_series = df['ret_1d']
    df['up_streak'] = _streak_count(ret_series > 0)
    df['down_streak'] = _streak_count(ret_series < 0)
    df['trend_persistence'] = df['up_streak'] - df['down_streak']

    # ── STEP 7: Volatility Regime ────────────────────────────
    df['rvol_20'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['rvol_60'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)
    df['vol_ratio'] = df['rvol_20'] / (df['rvol_60'] + 1e-8)
    df['vol_percentile'] = df['rvol_20'].rolling(252, min_periods=60).rank(pct=True)

    # ── STEP 8: Mean Reversion ───────────────────────────────
    ma20 = close_prev.rolling(20).mean()
    std20 = close_prev.rolling(20).std()
    df['bb_zscore'] = (close_prev - ma20) / (std20 + 1e-8)

    vwap_20 = (close_prev * volume_prev).rolling(20).sum() / \
              (volume_prev.rolling(20).sum() + 1e-8)
    df['vwap_dist'] = (close_prev - vwap_20) / (df['atr_14'] + 1e-8)

    # ── STEP 9: Momentum Quality ─────────────────────────────
    df['rsi_14'] = _compute_rsi(close_prev, 14)
    df['rsi_14_norm'] = (df['rsi_14'] - 50) / 50

    ema12 = close_prev.ewm(span=12, adjust=False).mean()
    ema26 = close_prev.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / (df['atr_14'] + 1e-8)
    df['macd_hist_norm'] = (macd - macd_signal) / (df['atr_14'] + 1e-8)

    # ── STEP 10: Higher-Order Statistics ─────────────────────
    df['skew_20'] = df['ret_1d'].rolling(20).skew()
    df['kurt_20'] = df['ret_1d'].rolling(20).kurt()

    # ── STEP 11: Market Structure ────────────────────────────
    rolling_high_20 = high_prev.rolling(20).max()
    rolling_low_20 = low_prev.rolling(20).min()
    price_range = rolling_high_20 - rolling_low_20
    df['dist_from_high'] = (rolling_high_20 - close_prev) / (price_range + 1e-8)
    df['dist_from_low'] = (close_prev - rolling_low_20) / (price_range + 1e-8)

    df['hh_streak'] = (high_prev > high_prev.shift(1)).astype(float).rolling(5).sum()
    df['ll_streak'] = (low_prev < low_prev.shift(1)).astype(float).rolling(5).sum()
    df['breakout_vol'] = (volume_prev > volume_prev.rolling(20).mean() * 1.5).astype(float)

    # ── STEP 12: Volume Regime ───────────────────────────────
    vol_mean = volume_prev.rolling(20).mean()
    vol_std = volume_prev.rolling(20).std()
    df['volume_zscore'] = (volume_prev - vol_mean) / (vol_std + 1e-8)
    df['vol_trend'] = volume_prev.rolling(5).mean() / (vol_mean + 1e-8)

    # ── STEP 13: Multi-Timeframe (simulated weekly) ──────────
    df['weekly_ret'] = close_prev.pct_change(5)
    df['weekly_trend'] = (close_prev.rolling(10).mean() > close_prev.rolling(40).mean()).astype(float)
    df['daily_weekly_align'] = (
        (df['ema_slope_20'] > 0) & (df['weekly_trend'] == 1)
    ).astype(float) - (
        (df['ema_slope_20'] < 0) & (df['weekly_trend'] == 0)
    ).astype(float)

    # ── STEP 14: Signal Quality ──────────────────────────────
    df['tradeable_trend'] = (df['adx'] > 20).astype(float)
    df['tradeable_vol'] = ((df['vol_percentile'] > 0.2) & (df['vol_percentile'] < 0.85)).astype(float)
    df['signal_quality'] = df['tradeable_trend'] * df['tradeable_vol']

    # ── STEP 15: Regime Detection ────────────────────────────
    df['regime_trending'] = ((df['ema_slope_20'] > 0.001) & (df['vol_percentile'] < 0.6)).astype(float)
    df['regime_volatile'] = ((df['vol_percentile'] > 0.8) | (df['ema_slope_20'] < -0.005)).astype(float)
    df['regime_sideways'] = (1 - df['regime_trending'] - df['regime_volatile']).clip(0, 1)

    # ── STEP 16: Advanced Features ───────────────────────────
    # Rolling Sharpe (using past returns only)
    roll_mean = df['ret_1d'].rolling(20).mean()
    roll_std = df['ret_1d'].rolling(20).std()
    df['rolling_sharpe_20'] = (roll_mean / (roll_std + 1e-8)) * np.sqrt(252)

    # Rolling drawdown
    cum_ret = (1 + df['ret_1d'].fillna(0)).cumprod()
    rolling_peak = cum_ret.rolling(60, min_periods=1).max()
    df['rolling_dd_60'] = (cum_ret - rolling_peak) / (rolling_peak + 1e-8)

    # Autocorrelation (lag-1)
    df['autocorr_5'] = df['ret_1d'].rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
    )

    # Hurst exponent approximation (R/S method, simplified)
    df['hurst_approx'] = df['ret_1d'].rolling(50, min_periods=20).apply(
        _hurst_rs, raw=True
    )

    # Volatility clustering (GARCH-like proxy)
    sq_ret = df['ret_1d'] ** 2
    df['vol_cluster'] = sq_ret.ewm(span=10, adjust=False).mean() / (sq_ret.rolling(50).mean() + 1e-8)

    # ── STEP 17: Target Engineering ──────────────────────────
    df = create_target(df, horizon=target_horizon,
                       threshold=target_threshold, method=target_method)

    # ── STEP 18: Cleanup ─────────────────────────────────────
    df = df.dropna(subset=['target']).reset_index(drop=True)
    # Drop rows where features are NaN (warm-up period)
    feature_cols = get_production_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"[FEAT-V4] {len(feature_cols)} features | {len(df):,} rows | "
          f"horizon={target_horizon}d | method={target_method} | "
          f"target_mean={df['target'].mean():.2%}")

    return df


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index using shift(1) data."""
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close = df['close'].shift(2)  # Previous close relative to shift(1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(period).mean()
    df['adx_norm'] = df['adx'] / 50

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _streak_count(condition: pd.Series) -> pd.Series:
    """Count consecutive True values. Resets on False."""
    groups = (~condition).cumsum()
    return condition.groupby(groups).cumsum().astype(float)


def _hurst_rs(data):
    """Simplified Hurst exponent via R/S analysis."""
    n = len(data)
    if n < 10:
        return 0.5
    mean = np.mean(data)
    y = np.cumsum(data - mean)
    r = np.max(y) - np.min(y)
    s = np.std(data, ddof=1)
    if s < 1e-10 or r < 1e-10:
        return 0.5
    return np.log(r / s) / np.log(n)


def get_production_feature_columns(df: pd.DataFrame) -> list:
    """
    Get clean feature column list.
    EXCLUDES: raw prices, target, future information, non-numeric.
    """
    exclude = {
        # Raw OHLCV — never use as features
        "date", "open", "high", "low", "close", "volume",
        # Target columns — future information
        "target", "target_return", "future_return",
        # Regime labels (string)
        "regime_state", "trend_regime_label", "vol_regime_label",
        "combined_regime_label",
    }
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def robust_normalize(df: pd.DataFrame, columns: List[str],
                     window: int = 100) -> pd.DataFrame:
    """
    Rolling Z-Score normalization (expanding, not lookahead).
    Uses expanding window with minimum periods to avoid lookahead.
    """
    df = df.copy()
    for col in columns:
        rolled = df[col].expanding(min_periods=20)
        df[f'{col}'] = (df[col] - rolled.mean()) / (rolled.std() + 1e-8)
    return df.fillna(0)
