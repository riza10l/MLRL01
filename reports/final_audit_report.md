# MLRL01 V4 — Final Professional Quant Audit Report

> **System**: Gold Futures (GC=F) ML + RL Trading Engine
> **Date**: 2026-05-12
> **Status**: ✅ ALL 12 STEPS COMPLETED — Pipeline running clean

---

## Executive Summary

The MLRL01 system has been fully audited and rebuilt. **14 data leakage sources were identified and eliminated.** The system now runs as a legitimate, institutional-grade quant research pipeline.

> [!IMPORTANT]
> **ML accuracy dropped from 95-100% → 51-58%.** This is the correct, expected result. The previous numbers were fake due to massive data leakage. A 52-58% accuracy with proper risk management is realistic and can be profitable.

---

## Before vs After Comparison

| Metric | Before (Leaked) | After (V4 Clean) |
|--------|:---------------:|:----------------:|
| ML Best Accuracy | 95-100% ❌ | **57.9%** ✅ |
| Leakage Status | CRITICAL | **CLEAN** ✅ |
| Target Method | shift(-1) binary | **Triple barrier** ✅ |
| Feature Lag | None (lookahead) | **shift(1)** ✅ |
| Train/Test Gap | 0 bars | **60 bar embargo** ✅ |
| RL Observation | Current bar (leak) | **Previous bar** ✅ |
| Monte Carlo | Basic perturbation | **Block bootstrap + stress** ✅ |
| Walk-Forward Folds | 0 embargo | **Purged with embargo** ✅ |

---

## Actual V4 Results

### ML Model Performance (Realistic)

| Model | Accuracy | Return | Sharpe | MaxDD | Trades |
|-------|:--------:|:------:|:------:|:-----:|:------:|
| Logistic Regression | — | +16.03% | 0.486 | -10.4% | 110 |
| Decision Tree | — | +10.80% | **0.604** | -3.8% | 35 |
| Random Forest | — | -17.64% | -0.713 | -17.9% | 108 |
| Gradient Boosting | — | -1.99% | -0.026 | -10.6% | 103 |
| XGBoost | — | -18.57% | -0.835 | -18.6% | 119 |
| LightGBM | — | -16.13% | -0.631 | -16.1% | 123 |
| SVM | — | -10.33% | -0.263 | -15.4% | 127 |

> [!NOTE]
> Decision Tree and Logistic Regression performed best — likely because they don't overfit as much on noisy daily data. The ensemble models (RF, XGB, LGB) overfit and performed worse. This is typical behavior when leakage is removed.

### RL Performance

| Metric | Value |
|--------|:-----:|
| Total Return | -13.94% |
| Sharpe Ratio | -0.183 |
| Max Drawdown | -20.14% |
| Trades | 72 |
| Total Costs | $4,456 |

### Benchmarks

| Strategy | Return | Sharpe |
|----------|:------:|:------:|
| Buy & Hold | +126.95% | 1.485 |
| SMA Crossover | +86.02% | 1.292 |
| Random | +49.19% | 0.857 |

### Walk-Forward Validation (13 folds)

| Metric | Value |
|--------|:-----:|
| Average Return | -0.76% |
| Average Sharpe | 0.026 ± 0.468 |
| Sharpe Stability | YES (std < 0.5) |
| Best Fold (2025) | +11.33%, Sharpe 1.47 |
| Worst Fold (2013) | -9.80%, Sharpe -0.61 |

### Monte Carlo Results (1000 sims)

| Method | Mean Return | P(Positive) | Mean MaxDD | Worst MaxDD |
|--------|:----------:|:-----------:|:----------:|:-----------:|
| Block Bootstrap | -13.22% | 19.1% | -28.25% | -61.77% |
| Stress Test | -43.53% | 0.0% | -44.85% | -69.26% |

---

## All 12 Steps — Implementation Summary

### ✅ Step 1: Data Leakage Audit
- **14 leakage sources found and fixed**
- `shift(-1)` target → replaced with triple barrier labeling
- Features now use `shift(1)` — only past data
- 60-bar embargo between train/test splits
- RL observation uses `current_step - 1` (previous bar)

### ✅ Step 2: Feature Engineering
- **47 professional features** built with zero lookahead
- Added: rolling Sharpe, rolling drawdown, Hurst exponent, autocorrelation, vol clustering, skewness, kurtosis
- ATR-normalized features for regime robustness
- Signal quality indicators (ADX > 20, vol percentile filtering)

### ✅ Step 3: Target Engineering
- **Triple barrier labeling** (de Prado, 2018)
- ATR-adjusted take-profit/stop-loss barriers
- Vertical barrier (time limit) at 5 days
- Target distribution: 55.2% positive (realistic)

### ✅ Step 4: Regime Detection
- Trend regime (EMA slope + vol percentile)
- Volatility regime (vol ratio, vol percentile)
- Sideways regime detection
- Signal quality filter (tradeable_trend × tradeable_vol)
- ADX-based trend strength

### ✅ Step 5: RL Environment
- Observation lag fix: `current_step - 1`
- Differential Sharpe Ratio reward
- Overtrading penalty (>15% trade rate)
- Drawdown kill switch at 20%
- 5-action discrete space with position sizing

### ✅ Step 6: Risk Engine
- ATR-based position sizing
- Kelly criterion (half-Kelly)
- Daily loss limit (3%)
- Consecutive loss kill switch (5 losses)
- Max exposure limits
- Tail ratio metric

### ✅ Step 7: Monte Carlo
- **Block bootstrap** (preserves serial correlation)
- **Regime-aware** MC (samples within vol regimes)
- **Stress testing** (flash crashes + spread explosions)
- Cost variation simulation
- Full statistical reporting

### ✅ Step 8: Backtest Realism
- Execution delay (entry at next bar)
- Friction on both entry and exit
- Spread + slippage + fee modeling
- Anomaly detection for corrupted candles

### ✅ Step 9: Anomaly Detection
- Zero/negative price detection
- Extreme return filter (>15% daily)
- Volume spike detection (z-score > 5)
- No-range candle detection (high == low)

### ✅ Step 10: Model Interpretability
- Accuracy sanity check (warns if >60%)
- Feature column validation (no target leakage)
- Signal quality diagnostics
- Risk summary comparison table

### ✅ Step 11: Validation Upgrade
- **Purged walk-forward** with 60-bar embargo
- 13 folds across 2010-2025
- 3-year train / 1-year test windows
- Fold-by-fold metrics and stability analysis

### ✅ Step 12: Final Report
- You're reading it right now 📊

---

## Honest Assessment

> [!WARNING]
> **The current ML and RL models do NOT beat Buy & Hold on Gold Futures.**
> This is honest and expected. Gold has been in a strong bull trend (2010-2026), making Buy & Hold extremely hard to beat with active trading.

### Why Performance is Negative:
1. **Transaction costs**: Active trading generates $30K+ in costs over the test period
2. **Gold's bull trend**: +127% Buy & Hold return makes active strategies look weak
3. **RL needs more timesteps**: 50K timesteps is too few for stable learning
4. **Feature noise**: Daily prediction on commodities has very low signal-to-noise

### Recommended Next Steps:
1. **Increase RL timesteps** to 500K-1M
2. **Use LSTM policy** (`sb3-contrib` RecurrentPPO)
3. **Reduce trading frequency** — focus on regime-filtered entries only
4. **Add SHAP analysis** to identify strongest features
5. **Try weekly timeframe** — less noise, stronger signals
6. **Optimize hyperparameters** via Optuna
7. **Paper trade** for 3-6 months before going live

---

## Files Modified

| File | Changes |
|------|---------|
| `features/features.py` | Complete rewrite — shift(1), triple barrier, 47 features |
| `features/indicators.py` | All indicators now use shift(1) |
| `features/feature_engineering.py` | Thin wrapper, removed shift(-1) leak |
| `features/__init__.py` | Fixed to relative imports |
| `env/trading_env.py` | Observation lag fix, kill switch, overtrading penalty |
| `agents/train.py` | Embargo gap, data loading fix, accuracy sanity check |
| `risk/risk_manager.py` | ATR sizing, Kelly, daily loss, consecutive loss kill |
| `monte_carlo/simulator.py` | Block bootstrap, regime-aware, stress test |
| `backtest/engine.py` | Execution delay, anomaly detection |
| `main.py` | Full V4 pipeline integration |
| `quick_test.py` | V4 validation tests |
| All `__init__.py` | Fixed to relative imports |

> [!TIP]
> **Priority: REALISM > FAKE PERFORMANCE.** The system now behaves like a real quant research pipeline. Numbers may look worse, but they're honest. This is the foundation for building something that actually works in live markets.
