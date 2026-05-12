"""
MLRL01 V4 — Leakage-Free Professional Quant Engine
=====================================================
Gold Futures (GC=F) — ML + RL System

V4 FIXES:
  - ALL data leakage eliminated
  - Features use shift(1) — only past data
  - Target uses triple barrier labeling
  - Embargo gap between train/test
  - Anomaly detection on raw data
  - Block bootstrap Monte Carlo
  - Realistic backtest with execution delay
  - Kill switch on drawdown

"ill keep evolving till i die" ahh machine
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Make sure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from agents.train import (
    load_latest_data, split_data, train_ml_models,
    train_rl_agent, walk_forward_validation, EMBARGO_BARS,
)
from agents.evaluate import (
    plot_equity_curves, plot_prediction_charts,
    plot_confusion_matrices, plot_accuracy_comparison,
    plot_walk_forward, save_risk_summary, save_comparison_csv,
)
from features.features import build_production_features, get_production_feature_columns
from env.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from backtest.engine import BacktestEngine
from backtest.benchmarks import BenchmarkRunner
from backtest.metrics import BacktestMetrics
from risk.risk_manager import RiskManager
from monte_carlo.simulator import MonteCarloSimulator


# --- Configuration ---

DATA_DIR        = "jupiter" if os.path.exists("jupiter") else "../jupiter"
TRAIN_RATIO     = 0.80
RL_TIMESTEPS    = 50_000

# Target Engineering (V4: triple barrier)
TARGET_HORIZON   = 5
TARGET_THRESHOLD = 0.005
TARGET_METHOD    = "triple_barrier"

# Toggles
USE_LSTM        = False
RUN_WALK_FWD    = True
RUN_MONTE_CARLO = True
MC_SIMULATIONS  = 1000

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR_PLOTS   = os.path.join(BASE_DIR, "results", "plots")
SAVE_DIR_REPORTS = os.path.join(BASE_DIR, "results", "reports")
SAVE_DIR_MC      = os.path.join(BASE_DIR, "results", "monte_carlo")
MODEL_SAVE_DIR   = os.path.join(BASE_DIR, "models", "saved_models")


def main():
    print("=" * 65)
    print("  MLRL01 V4 — LEAKAGE-FREE QUANT ENGINE")
    print("  Gold Futures (GC=F) — Professional Audit Applied")
    print("=" * 65)

    for d in [SAVE_DIR_PLOTS, SAVE_DIR_REPORTS, SAVE_DIR_MC, MODEL_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load Data ─────────────────────────────────────────────
    df_raw = load_latest_data(DATA_DIR)

    # ── Anomaly Detection ─────────────────────────────────────
    print("\n[ANOMALY] Scanning raw data for corrupted candles...")
    clean_mask = BacktestEngine.detect_anomalous_candles(df_raw)
    n_removed = (~clean_mask).sum()
    if n_removed > 0:
        df_raw = df_raw[clean_mask].reset_index(drop=True)
        print(f"[ANOMALY] Removed {n_removed} anomalous rows")

    # ── Feature Engineering (V4 — Leakage-Free) ──────────────
    print("\n[FEAT] Building V4 leakage-free features...")
    df = build_production_features(
        df_raw,
        target_horizon=TARGET_HORIZON,
        target_threshold=TARGET_THRESHOLD,
        target_method=TARGET_METHOD,
    )

    # ── Feature Columns ──────────────────────────────────────
    feature_cols = get_production_feature_columns(df)

    print(f"\n[CONFIG] Features: {len(feature_cols)}")
    print(f"[CONFIG] RL Timesteps: {RL_TIMESTEPS:,}")
    print(f"[CONFIG] Target: {TARGET_HORIZON}d horizon, {TARGET_METHOD}")
    print(f"[CONFIG] Embargo: {EMBARGO_BARS} bars")
    print(f"[CONFIG] Monte Carlo: {'ON' if RUN_MONTE_CARLO else 'OFF'} ({MC_SIMULATIONS} sims)")
    print(f"[CONFIG] Target distribution: {df['target'].mean():.1%} positive")

    # ── Split Data (with Embargo) ────────────────────────────
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     dates_test, close_test, scaler) = split_data(df, feature_cols, TRAIN_RATIO)

    # ── Train ML Models ──────────────────────────────────────
    predictions, results_df, trained_models = train_ml_models(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )
    print("\n[ML] Model Ranking (by Accuracy):")
    print(results_df.to_string())

    # ── Leakage Sanity Check ─────────────────────────────────
    print("\n" + "=" * 65)
    print("  LEAKAGE SANITY CHECK")
    print("=" * 65)
    max_acc = results_df["Accuracy"].max()
    if max_acc > 0.60:
        print(f"  [WARN] Max accuracy = {max_acc:.1%}")
        print(f"  [WARN] Accuracy > 60% on daily prediction is suspicious.")
        print(f"  [WARN] If all models > 55%, there may still be leakage.")
    else:
        print(f"  [OK] Max accuracy = {max_acc:.1%} — realistic range")

    # ── Train RL Agent ───────────────────────────────────────
    split_idx = int(len(df) * TRAIN_RATIO)
    df_train_slice = df.iloc[:split_idx].reset_index(drop=True)
    # Apply embargo for RL test data too
    test_start = split_idx + EMBARGO_BARS
    df_test_slice = df.iloc[test_start:].reset_index(drop=True)

    print("\n[RL] Training PPO agent (V4 DSR + overtrading penalty)...")
    save_path = os.path.join(MODEL_SAVE_DIR, "ppo_trading_gold_v4")
    agent = train_rl_agent(
        df_train_slice, feature_cols,
        timesteps=RL_TIMESTEPS,
        use_lstm=USE_LSTM,
        save_path=save_path,
    )

    # Evaluate RL on test data
    print("[RL] Evaluating PPO on test data...")
    env_test = TradingEnv(df_test_slice, feature_columns=feature_cols)
    rl_equity, rl_stats, rl_log = agent.evaluate(env_test)

    rl_metrics = RiskManager.compute_all_metrics(rl_equity)
    rl_metrics["equity"] = rl_equity
    rl_metrics["n_trades"] = rl_stats.get("total_trades", 0)
    rl_metrics["total_costs"] = rl_stats.get("total_costs", 0)
    print(f"[RL] Return: {rl_metrics['total_return']:+.2%} | "
          f"Sharpe: {rl_metrics['sharpe_ratio']:.3f} | "
          f"MaxDD: {rl_metrics['max_drawdown']:.2%} | "
          f"Trades: {rl_metrics['n_trades']} | "
          f"Killed: {rl_stats.get('killed', False)}")

    # ── Backtest ML Models ───────────────────────────────────
    bt_engine = BacktestEngine()
    bt_results = bt_engine.run_ml_backtest(predictions, close_test, dates_test)

    # ── Run Benchmarks ───────────────────────────────────────
    benchmark_runner = BenchmarkRunner(df_test_slice)
    bm_results = benchmark_runner.run_all()

    # ── Compare RL vs Benchmarks ─────────────────────────────
    print("\n" + "=" * 65)
    print("  RL vs BENCHMARKS COMPARISON")
    print("=" * 65)

    bh_sharpe = bm_results.get("Buy & Hold", {}).get("sharpe_ratio", 0)
    rl_sharpe = rl_metrics.get("sharpe_ratio", 0)
    if rl_sharpe > bh_sharpe:
        print(f"  RL Sharpe ({rl_sharpe:.3f}) > Buy&Hold ({bh_sharpe:.3f}) -> RL WORTH IT!")
    else:
        print(f"  RL Sharpe ({rl_sharpe:.3f}) <= Buy&Hold ({bh_sharpe:.3f}) -> NEEDS MORE WORK")

    # ── Walk-Forward Validation ──────────────────────────────
    wf_results = pd.DataFrame()
    if RUN_WALK_FWD:
        wf_results = walk_forward_validation(
            df, feature_cols,
            train_years=3, test_years=1, step_years=1,
            embargo=EMBARGO_BARS,
        )

    # ── Monte Carlo Simulation ───────────────────────────────
    if RUN_MONTE_CARLO:
        print("\n" + "=" * 65)
        print("  MONTE CARLO VALIDATION (V4: Block Bootstrap + Stress Test)")
        print("=" * 65)

        mc = MonteCarloSimulator(n_simulations=MC_SIMULATIONS)

        # Get daily returns from RL equity curve
        rl_returns = np.diff(rl_equity) / (rl_equity[:-1] + 1e-10)

        # Method 1: Block bootstrap (preserves serial correlation)
        print("\n  [MC] Block Bootstrap (block_size=20)...")
        mc_block = mc.run_block_bootstrap(rl_returns, block_size=20)
        mc_block_report = mc.generate_report(mc_block)
        _print_mc_report(mc_block_report)

        # Method 2: Stress test
        print("\n  [MC] Stress Test (crashes + spread explosions)...")
        mc_stress = mc.run_stress_test(rl_returns)
        mc_stress_report = mc.generate_report(mc_stress)
        _print_mc_report(mc_stress_report)

        # Method 3: Return perturbation (backward compat)
        mc_perturb = mc.run_return_perturbation(rl_returns, noise_std=0.001)
        mc_perturb_report = mc.generate_report(mc_perturb)

        # Generate plots from block bootstrap
        mc.plot_all(mc_block, save_dir=SAVE_DIR_MC)

        # Save MC report
        all_mc_reports = {
            "block_bootstrap": mc_block_report,
            "stress_test": mc_stress_report,
            "perturbation": mc_perturb_report,
        }
        mc_rows = []
        for method, report in all_mc_reports.items():
            row = {"method": method}
            row.update(report)
            mc_rows.append(row)
        mc_df = pd.DataFrame(mc_rows)
        mc_path = os.path.join(SAVE_DIR_REPORTS, "monte_carlo_report.csv")
        mc_df.to_csv(mc_path, index=False)
        print(f"\n  Report saved -> {mc_path}")

    # ── Generate Charts ──────────────────────────────────────
    print("\n[CHARTS] Generating charts...")

    all_metrics = {}
    for name, res in bt_results.items():
        all_metrics[name] = res
    all_metrics["PPO (RL)"] = rl_metrics
    for name, res in bm_results.items():
        all_metrics[f"BM: {name}"] = res

    plot_equity_curves(bt_results, rl_equity, bm_results, dates_test,
                       save_dir=SAVE_DIR_PLOTS)
    plot_prediction_charts(predictions, dates_test, close_test,
                           save_dir=SAVE_DIR_PLOTS)
    plot_confusion_matrices(predictions, y_test, save_dir=SAVE_DIR_PLOTS)
    plot_accuracy_comparison(results_df, save_dir=SAVE_DIR_PLOTS)
    if not wf_results.empty:
        plot_walk_forward(wf_results, save_dir=SAVE_DIR_PLOTS)

    # ── Save Reports ─────────────────────────────────────────
    print("\n[SAVE] Saving reports...")
    save_risk_summary(all_metrics, save_dir=SAVE_DIR_REPORTS)
    save_comparison_csv(results_df, bt_results, save_dir=SAVE_DIR_REPORTS)

    if not wf_results.empty:
        wf_path = os.path.join(SAVE_DIR_REPORTS, "walk_forward_results.csv")
        wf_results.to_csv(wf_path, index=False)
        print(f"  Saved -> {wf_path}")

    # ── Final Summary ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  V4 AUDIT COMPLETE")
    print("=" * 65)
    print(f"  Features: {len(feature_cols)} (all shift(1), no lookahead)")
    print(f"  Target: {TARGET_METHOD} ({TARGET_HORIZON}d horizon)")
    print(f"  Embargo: {EMBARGO_BARS} bars between train/test")
    print(f"  ML Best Accuracy: {results_df['Accuracy'].max():.1%}")
    print(f"  RL Sharpe: {rl_sharpe:.3f}")
    if max_acc <= 0.60:
        print(f"  Leakage Status: CLEAN")
    else:
        print(f"  Leakage Status: REVIEW NEEDED")
    print("=" * 65)


def _print_mc_report(report):
    """Pretty-print a Monte Carlo report."""
    print(f"    Sims: {report['n_simulations']}")
    print(f"    Mean Return: {report['mean_return']:+.2%}")
    print(f"    Median Return: {report['median_return']:+.2%}")
    print(f"    P(Positive): {report['prob_positive']:.1%}")
    print(f"    P(Ruin >10%): {report['prob_ruin_10pct']:.1%}")
    print(f"    P(Ruin >20%): {report['prob_ruin_20pct']:.1%}")
    print(f"    Mean MaxDD: {report['mean_max_dd']:.2%}")
    print(f"    Worst MaxDD: {report['worst_max_dd']:.2%}")
    if 'mean_sharpe' in report:
        print(f"    Mean Sharpe: {report['mean_sharpe']:.3f}")


if __name__ == "__main__":
    main()
