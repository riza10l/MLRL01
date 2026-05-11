# 🚀 MLRL01 Monster Mode: Final Integration & Deployment Guide

This document summarizes the final wiring and verification steps for the upgraded Reinforcement Learning trading system.

## 🔗 System Wiring Diagram

The pipeline flows as follows:

1.  **`features.py`**: Ingests raw OHLCV data from `jupiter/dataXX.csv`. Computes 20+ rich indicators and classifies the market into 3 regimes (Sideways, Bullish, Crisis).
2.  **`env_trading.py`**: Wraps the processed data into a Gymnasium environment. Handles the 5-action space (Long/Short/Flat), enforces `min_hold_period`, and calculates the risk-adjusted reward ($r = ret - \lambda_{vol} \cdot \sigma - \lambda_{dd} \cdot DD$).
3.  **`train_wf.py`**: The orchestration script. It slides a 4-year window (3y train / 1y test) across history, training a **RecurrentPPO (LSTM)** agent in each fold and benchmarking it against Buy & Hold.

## 🛠️ Execution Checklist

### 1. Dry-Run Verification
Ensure the environment and feature shapes are consistent.
```powershell
# Quick check of the new environment and actions
python MLRL01/quick_test.py
```

### 2. Full Walk-Forward Training
Run the production pipeline with LSTM architecture.
```powershell
# Recommended production run
python MLRL01/train_wf.py --timesteps 100000 --train_years 3 --test_years 1
```

### 3. Check Seed Reproducibility
The `config.yaml` sets `seed: 42`. Running the same command twice should yield identical results in the `logs/` directory.

## 📊 Metric Targets
Monitor the `results/wf_v3/wf_summary.csv` for these goals:
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 10%
- **Win Rate**: 45–60%
- **WFV Stability**: Sharpe deviation across folds < 0.5

## 🛡️ Production & Graceful Degradation

### Async & Memory Management
- **LSTM Memory**: The `seq_len` is capped at 24 bars. For higher timeframes (e.g., Weekly), reduce this to 12 to avoid memory fragmentation.
- **Graceful Fallback**: If `RecurrentPPO` fails to converge (detected by flat equity), the system falls back to the **Regime-based SMA** logic defined in Phase 2.

### Handling Missing Data
The `features.py` pipeline uses `dropna()` after all rolling computations. This ensures the agent never sees a `NaN`, but it also means the first 100-250 bars of history (the warmup period) are discarded. Ensure your raw CSV has at least 1000+ rows.

---
**Status**: All components (P1-P12) are now compiled and integrated. The system is ready for automated daily walk-forward execution.
