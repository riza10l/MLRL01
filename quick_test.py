"""
Quick Test Script (V3)
=======================
Test the V3 TradingEnv, features, and Monte Carlo without full training.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.features import build_production_features, get_production_feature_columns
from env.trading_env import TradingEnv
from agents.train import load_latest_data
from monte_carlo.simulator import MonteCarloSimulator

def test_env():
    data_dir = "jupiter" if os.path.exists("jupiter") else "../jupiter"
    df_raw = load_latest_data(data_dir)
    
    print("[TEST] Building V3 features (5d horizon, 0.5% threshold)...")
    df = build_production_features(df_raw, target_horizon=5, target_threshold=0.005)
    
    feature_cols = get_production_feature_columns(df)
    print(f"[TEST] Feature columns: {len(feature_cols)}")
    print(f"[TEST] Sample features: {feature_cols[:5]}...")
    
    # Check signal quality distribution
    sq = df['signal_quality'].mean()
    adx_mean = df['adx'].mean()
    print(f"[TEST] Signal quality avg: {sq:.2%} (% of bars tradeable)")
    print(f"[TEST] ADX avg: {adx_mean:.1f}")
    
    print("\n[TEST] Initializing TradingEnv V3...")
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
    
    # Quick Monte Carlo test
    print("\n[TEST] Quick Monte Carlo (100 sims)...")
    returns = np.diff(eq) / (eq[:-1] + 1e-10)
    mc = MonteCarloSimulator(n_simulations=100, seed=42)
    result = mc.run_return_perturbation(returns, noise_std=0.001)
    report = mc.generate_report(result)
    print(f"  MC Mean Return: {report['mean_return']:+.2%}")
    print(f"  MC P(Positive): {report['prob_positive']:.1%}")
    print(f"  MC Mean MaxDD: {report['mean_max_dd']:.2%}")
    
    print("\n[TEST] V3 system works! [DONE]")

if __name__ == "__main__":
    test_env()
