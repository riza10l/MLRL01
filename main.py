"""
MLRL01 V3 - Professional Quant Trading Engine
================================================
Gold Futures (GC=F) — ML + RL System
HM = Riza Wahyu Nugraha

V3 Upgrades:
  - Multi-horizon target labels (5-day with threshold)
  - ADX + trend persistence features
  - Signal quality filters
  - DSR reward function
  - Multi-layer overtrading control
  - Monte Carlo validation
  - Professional metrics

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
    train_rl_agent, walk_forward_validation,
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


# ══════════════════════════════════════════════════════════════
#  V3 CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_DIR        = "jupiter" if os.path.exists("jupiter") else "../jupiter"
TRAIN_RATIO     = 0.80
RL_TIMESTEPS    = 50_000

# Target Engineering
TARGET_HORIZON   = 5       # 5-day forward return
TARGET_THRESHOLD = 0.005   # 0.5% minimum move

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
    print("  MLRL01 V3 TRADING ENGINE - Gold Futures (GC=F)")
    print("  Professional Quant Upgrade: DSR + MC + Signal Quality")
    print("=" * 65)

    for d in [SAVE_DIR_PLOTS, SAVE_DIR_REPORTS, SAVE_DIR_MC, MODEL_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────
    df_raw = load_latest_data(DATA_DIR)

    # ── 2. V3 Feature Engineering ─────────────────────────────
    df = build_production_features(
        df_raw, 
        target_horizon=TARGET_HORIZON, 
        target_threshold=TARGET_THRESHOLD
    )

    # ── 3. Get feature columns ────────────────────────────────
    feature_cols = get_production_feature_columns(df)
    # Fallback for ML training (needs old feature_engineering module)
    try:
        from features.feature_engineering import get_feature_columns
        ml_feature_cols = get_feature_columns(df)
        if not ml_feature_cols:
            ml_feature_cols = feature_cols
    except Exception:
        ml_feature_cols = feature_cols

    print(f"\n[CONFIG] Features: {len(feature_cols)}")
    print(f"[CONFIG] RL Timesteps: {RL_TIMESTEPS:,}")
    print(f"[CONFIG] Target: {TARGET_HORIZON}d horizon, {TARGET_THRESHOLD:.1%} threshold")
    print(f"[CONFIG] Monte Carlo: {'ON' if RUN_MONTE_CARLO else 'OFF'} ({MC_SIMULATIONS} sims)")

    # ── 4. Split Data ─────────────────────────────────────────
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     dates_test, close_test, scaler) = split_data(df, ml_feature_cols, TRAIN_RATIO)

    # ── 5. Train ML Models ────────────────────────────────────
    predictions, results_df, trained_models = train_ml_models(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )
    print("\n[ML] Model Ranking (by Accuracy):")
    print(results_df.to_string())

    # ── 6. Train RL Agent ─────────────────────────────────────
    split_idx = int(len(df) * TRAIN_RATIO)
    df_train_slice = df.iloc[:split_idx].reset_index(drop=True)
    df_test_slice = df.iloc[split_idx:].reset_index(drop=True)

    print("\n[RL] Training PPO agent (V3 DSR reward)...")
    save_path = os.path.join(MODEL_SAVE_DIR, "ppo_trading_gold_v3")
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
          f"Trades: {rl_metrics['n_trades']}")

    # ── 7. Backtest ML Models ─────────────────────────────────
    bt_engine = BacktestEngine()
    bt_results = bt_engine.run_ml_backtest(predictions, close_test, dates_test)

    # ── 8. Run Benchmarks ─────────────────────────────────────
    benchmark_runner = BenchmarkRunner(df_test_slice)
    bm_results = benchmark_runner.run_all()

    # ── 9. Compare RL vs Benchmarks ───────────────────────────
    print("\n" + "=" * 65)
    print("  RL vs BENCHMARKS COMPARISON")
    print("=" * 65)

    bh_sharpe = bm_results.get("Buy & Hold", {}).get("sharpe_ratio", 0)
    rl_sharpe = rl_metrics.get("sharpe_ratio", 0)
    if rl_sharpe > bh_sharpe:
        print(f"  RL Sharpe ({rl_sharpe:.3f}) > Buy&Hold ({bh_sharpe:.3f}) -> RL WORTH IT!")
    else:
        print(f"  RL Sharpe ({rl_sharpe:.3f}) <= Buy&Hold ({bh_sharpe:.3f}) -> NEEDS MORE WORK")

    # ── 10. Walk-Forward Validation ───────────────────────────
    wf_results = pd.DataFrame()
    if RUN_WALK_FWD:
        wf_results = walk_forward_validation(
            df, feature_cols,
            train_years=2, test_years=1, step_years=1,
        )

    # ── 11. Monte Carlo Simulation ────────────────────────────
    if RUN_MONTE_CARLO:
        print("\n" + "=" * 65)
        print("  MONTE CARLO VALIDATION")
        print("=" * 65)
        
        mc = MonteCarloSimulator(n_simulations=MC_SIMULATIONS)
        
        # Get daily returns from RL equity curve
        rl_returns = np.diff(rl_equity) / (rl_equity[:-1] + 1e-10)
        
        # Method 1: Return perturbation
        mc_result = mc.run_return_perturbation(rl_returns, noise_std=0.001)
        mc_report = mc.generate_report(mc_result)
        
        print(f"  Simulations: {mc_report['n_simulations']}")
        print(f"  Mean Return: {mc_report['mean_return']:+.2%}")
        print(f"  Median Return: {mc_report['median_return']:+.2%}")
        print(f"  P(Positive): {mc_report['prob_positive']:.1%}")
        print(f"  P(Ruin >10%): {mc_report['prob_ruin_10pct']:.1%}")
        print(f"  P(Ruin >20%): {mc_report['prob_ruin_20pct']:.1%}")
        print(f"  Mean MaxDD: {mc_report['mean_max_dd']:.2%}")
        print(f"  Worst MaxDD: {mc_report['worst_max_dd']:.2%}")
        if 'mean_sharpe' in mc_report:
            print(f"  Mean Sharpe: {mc_report['mean_sharpe']:.3f}")
            print(f"  Sharpe 5th-95th: [{mc_report['sharpe_5th_pct']:.3f}, {mc_report['sharpe_95th_pct']:.3f}]")
        
        # Generate plots
        mc.plot_all(mc_result, save_dir=SAVE_DIR_MC)
        
        # Save MC report
        mc_df = pd.DataFrame([mc_report])
        mc_path = os.path.join(SAVE_DIR_REPORTS, "monte_carlo_report.csv")
        mc_df.to_csv(mc_path, index=False)
        print(f"  Report saved -> {mc_path}")

    # ── 12. Generate Charts ───────────────────────────────────
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

    # ── 13. Save Reports ──────────────────────────────────────
    print("\n[SAVE] Saving reports...")
    save_risk_summary(all_metrics, save_dir=SAVE_DIR_REPORTS)
    save_comparison_csv(results_df, bt_results, save_dir=SAVE_DIR_REPORTS)

    if not wf_results.empty:
        wf_path = os.path.join(SAVE_DIR_REPORTS, "walk_forward_results.csv")
        wf_results.to_csv(wf_path, index=False)
        print(f"  Saved -> {wf_path}")

    print("\n" + "=" * 65)
    print("  SELESAI! Check results/ folder buat semua output.")
    print("  V3: DSR reward + Signal Quality + Monte Carlo")
    print("=" * 65)


if __name__ == "__main__":
    main()
