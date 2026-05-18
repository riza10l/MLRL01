# MLRL01 — Full Professional Quant Audit Report

> **System**: Gold Futures (GC=F) ML + RL Trading Engine
> **Date**: 2026-05-12
> **Auditor**: Senior Quant Researcher

---

## Executive Summary

The MLRL01 system exhibits **critical data leakage** across multiple layers, producing artificially inflated ML accuracy (95-100%). After fixing leakage, realistic accuracy should be **50-55%** — which is actually profitable if paired with proper risk management and edge sizing.

> [!CAUTION]
> **14 data leakage sources identified.** The current system cannot be trusted for any trading decisions until all leakage is eliminated.

---

## STEP 1 — Data Leakage Audit Findings

### Finding #1: Target Leakage via `shift(-1)` — CRITICAL
- **File**: [feature_engineering.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/features/feature_engineering.py#L52)
- **Code**: `df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)`
- **Problem**: `shift(-1)` uses tomorrow's close to create today's label. When combined with features computed on today's close, the model can trivially learn patterns that encode the target direction.
- **Danger Level**: 🔴 CRITICAL — This alone explains 95%+ accuracy
- **Fix**: Use multi-day forward return with proper gap. Target should be `close[t+horizon] > close[t]` with embargo period.

### Finding #2: Target Leakage in `features.py` — CRITICAL
- **File**: [features.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/features/features.py#L20)
- **Code**: `df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)`
- **Problem**: `future_return` column persists in DataFrame. Even though `target` is derived from it, the column itself leaks future information into any model that accesses the full DataFrame.
- **Danger Level**: 🔴 CRITICAL
- **Fix**: Drop `future_return` and `target_return` from feature columns. Already partially handled by `get_production_feature_columns()` exclude list, but must be verified at every pipeline stage.

### Finding #3: Scaler Fitted on Full Dataset — HIGH
- **File**: [train.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/agents/train.py#L60-L62)
- **Code**: `scaler = StandardScaler(); X_train_sc = scaler.fit_transform(X_train)`
- **Status**: ✅ Actually correct — scaler is fit on train only
- **But**: The `split_data()` function does a simple 80/20 split without any **embargo/purge gap**. Rows at the boundary share rolling windows with test data.

### Finding #4: Rolling Feature Contamination — HIGH
- **Files**: [indicators.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/features/indicators.py), [features.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/features/features.py)
- **Problem**: Rolling features (MA50, rolling_vol_20, etc.) are computed on the entire dataset before train/test split. The test set's rolling windows overlap with training data — this is **expected and acceptable** for rolling indicators. However, the features are computed using `close` at time `t` without any `shift(1)` lag.
- **Danger Level**: 🟡 MEDIUM — Features use today's close to predict today's target. Institutional-grade systems use `shift(1)` to ensure all features only use data available at decision time (market open).
- **Fix**: Add `shift(1)` to all features that use current-bar close/volume.

### Finding #5: Walk-Forward Validation Has No Embargo — HIGH
- **File**: [train.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/agents/train.py#L128-L136)
- **Problem**: Train/test folds are split by year boundary with no purge/embargo gap. Rolling indicators from the last N bars of training contaminate the first N bars of testing.
- **Fix**: Add embargo of `max(rolling_windows)` bars between train and test folds.

### Finding #6: RL Environment Uses Current-Bar Features — MEDIUM
- **File**: [trading_env.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/env/trading_env.py#L217)
- **Code**: `latest = self.df.iloc[self.current_step][self.feature_columns]`
- **Problem**: The RL agent observes features at `current_step`, then the `step()` function computes reward based on the same bar's close price. This creates a subtle lookahead where the agent sees the bar's features (which include that bar's close) before the bar completes.
- **Fix**: Observe features at `current_step - 1`, act at `current_step`.

### Finding #7: `dropna()` After Feature Build Causes Index Misalignment — LOW
- **Files**: Multiple
- **Problem**: After `dropna()`, index is reset. But date alignment with original data may be lost, causing off-by-one errors in backtest.
- **Fix**: Maintain date column as primary key.

### Finding #8: ML Backtest Uses Same-Day Price — MEDIUM
- **File**: [engine.py](file:///c:/Users/Riza%20Wahyu%20Nugraha/Documents/coolyeah/code/external/trainszzz/MLRL01/backtest/engine.py#L42-L43)
- **Code**: `price = closes[i]; price_next = closes[i + 1]`
- **Problem**: Prediction at bar `i` uses bar `i`'s features (which include bar `i`'s close). Entry happens at `price_next`, but the decision already incorporated `price` info.
- **Fix**: With proper feature lagging (`shift(1)`), this becomes acceptable.

### Finding #9: Monte Carlo Uses Raw Equity Returns — LOW
- **Problem**: Monte Carlo simulation on daily equity returns doesn't account for serial correlation or regime structure.
- **Fix**: Implement block bootstrap and regime-aware MC.

### Finding #10: No Feature Importance Validation — MEDIUM
- **Problem**: No check for features that are proxies for the target.
- **Fix**: Add SHAP analysis and permutation importance.

### Finding #11: Naive Target (Next-Bar Binary) — HIGH
- **Problem**: Binary next-bar prediction is dominated by market microstructure noise. Signal-to-noise ratio is extremely low.
- **Fix**: Triple barrier labeling with volatility-adjusted thresholds.

### Finding #12: No Purged Cross-Validation — HIGH
- **Problem**: Standard train/test split allows information bleed through overlapping labels.
- **Fix**: Implement purged k-fold with embargo.

### Finding #13: Feature `sma_cross_signal` Not Excluded Properly — LOW
- **Problem**: This is a non-numeric label in some code paths but included in others.
- **Fix**: Consistently exclude from feature columns.

### Finding #14: Multiple Feature Pipelines — ARCHITECTURE
- **Problem**: `feature_engineering.py` and `features.py` are two separate feature pipelines. `main.py` uses `features.py`, `train.py` uses `feature_engineering.py`. This creates confusion about which features are used where.
- **Fix**: Consolidate into single pipeline.

---

## Implementation Plan — 12 Steps

### Phase 1: Critical Fixes (Steps 1-3)

| Step | Priority | Description |
|------|----------|-------------|
| 1 | 🔴 CRITICAL | Fix all data leakage — feature lagging, target engineering, embargo |
| 2 | 🔴 CRITICAL | Consolidate feature pipeline — single source of truth |
| 3 | 🔴 HIGH | Professional target engineering — triple barrier labeling |

### Phase 2: Architecture Upgrades (Steps 4-6)

| Step | Priority | Description |
|------|----------|-------------|
| 4 | 🟡 HIGH | Regime detection with HMM/clustering |
| 5 | 🟡 HIGH | RL environment improvement — reward, state, action space |
| 6 | 🟡 HIGH | Professional risk engine |

### Phase 3: Validation & Robustness (Steps 7-9)

| Step | Priority | Description |
|------|----------|-------------|
| 7 | 🟡 MEDIUM | Monte Carlo upgrades — block bootstrap, regime-aware |
| 8 | 🟡 MEDIUM | Backtest realism — slippage, spread, execution delay |
| 9 | 🟡 MEDIUM | Anomaly detection |

### Phase 4: Analysis & Reporting (Steps 10-12)

| Step | Priority | Description |
|------|----------|-------------|
| 10 | 🔵 LOW | Model interpretability — SHAP, feature importance |
| 11 | 🔵 LOW | Validation methodology upgrades |
| 12 | 🔵 LOW | Final professional report |

---

> [!IMPORTANT]
> After fixing leakage, ML accuracy will drop from 95%+ to ~50-55%. **This is correct and expected.** A 52-55% accuracy with proper risk management and 1:2 risk-reward ratio is actually profitable. The current 95%+ is fake.

---

## Expected Realistic Performance After Fixes

| Metric | Before (Fake) | After (Real) |
|--------|--------------|--------------|
| ML Accuracy | 95-100% | 50-55% |
| Sharpe Ratio | Inflated | 0.3-0.8 |
| Max Drawdown | Understated | 15-30% |
| Win Rate | 90%+ | 48-55% |
| Annual Return | Unrealistic | 5-15% |

These are realistic numbers for a Gold Futures daily strategy. Anything claiming >60% accuracy on daily bars is almost certainly leaking.