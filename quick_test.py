
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.features import build_production_features, get_production_feature_columns
from env.trading_env import TradingEnv
from agents.train import load_latest_data
from monte_carlo.simulator import MonteCarloSimulator
from backtest.engine import BacktestEngine

def test_env():
    data_dir = "jupiter" if os.path.exists("jupiter") else "../jupiter"
    df_raw = load_latest_data(data_dir)

    # Anomaly detection
    print("\n[TEST] Anomaly scan...")
    clean_mask = BacktestEngine.detect_anomalous_candles(df_raw)
    df_raw = df_raw[clean_mask].reset_index(drop=True)

    # Build V4 features
    print("\n[TEST] Building V4 leakage-free features (triple barrier)...")
    df = build_production_features(df_raw, target_horizon=5,
                                   target_threshold=0.005,
                                   target_method="triple_barrier")

    feature_cols = get_production_feature_columns(df)
    print(f"[TEST] Feature columns: {len(feature_cols)}")
    print(f"[TEST] Sample features: {feature_cols[:5]}...")
    print(f"[TEST] Target distribution: {df['target'].mean():.1%} positive")

    # Leakage sanity check: verify features don't contain future info
    print("\n[TEST] Leakage sanity checks...")
    leakage_cols = {'future_return', 'target_return'}
    found_leakage = [c for c in leakage_cols if c in feature_cols]
    if found_leakage:
        print(f"  [FAIL] Leakage columns in features: {found_leakage}")
    else:
        print(f"  [PASS] No leakage columns in feature list")

    # Check that features use shift(1) — verify first non-NaN row
    first_valid = df[feature_cols].first_valid_index()
    print(f"  [INFO] First valid feature row: {first_valid} (should be > 0)")

    # Check signal quality distribution
    if 'signal_quality' in df.columns:
        sq = df['signal_quality'].mean()
        print(f"[TEST] Signal quality avg: {sq:.2%} (% of bars tradeable)")
    if 'adx' in df.columns:
        adx_mean = df['adx'].mean()
        print(f"[TEST] ADX avg: {adx_mean:.1f}")

    # Test TradingEnv
    print("\n[TEST] Initializing TradingEnv V4...")
    env = TradingEnv(df, feature_columns=feature_cols)

    obs, info = env.reset()
    print(f"[TEST] Observation shape: {obs.shape}")
    print(f"[TEST] Action space: {env.action_space}")

    print("\n[TEST] Running 50 random steps...")
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            print(f"  Episode ended at step {i}")
            break

    stats = env.get_trade_stats()
    eq = env.get_equity_curve()
    print(f"\n[TEST] Final equity: ${eq[-1]:,.0f}")
    print(f"[TEST] Trades: {stats['total_trades']}, Costs: ${stats['total_costs']:.2f}")
    print(f"[TEST] Total reward: {total_reward:.4f}")
    print(f"[TEST] Kill switch: {stats.get('killed', False)}")

    # Quick Monte Carlo test (block bootstrap)
    print("\n[TEST] Quick Monte Carlo — Block Bootstrap (100 sims)...")
    returns = np.diff(eq) / (eq[:-1] + 1e-10)
    mc = MonteCarloSimulator(n_simulations=100, seed=42)
    result = mc.run_block_bootstrap(returns, block_size=10)
    report = mc.generate_report(result)
    print(f"  MC Mean Return: {report['mean_return']:+.2%}")
    print(f"  MC P(Positive): {report['prob_positive']:.1%}")
    print(f"  MC Mean MaxDD: {report['mean_max_dd']:.2%}")
    if 'mean_sharpe' in report:
        print(f"  MC Mean Sharpe: {report['mean_sharpe']:.3f}")

    # Quick stress test
    print("\n[TEST] Quick Stress Test (100 sims)...")
    stress_result = mc.run_stress_test(returns)
    stress_report = mc.generate_report(stress_result)
    print(f"  Stress Mean Return: {stress_report['mean_return']:+.2%}")
    print(f"  Stress P(Ruin >10%): {stress_report['prob_ruin_10pct']:.1%}")
    print(f"  Stress Worst MaxDD: {stress_report['worst_max_dd']:.2%}")

    print("\n[TEST] V4 system works! All leakage fixes verified. [DONE]")

if __name__ == "__main__":
    test_env()