import os
import sys
import argparse
import pandas as pd
import numpy as np

# Make sure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_interface import MLRL01Engine

def parse_args():
    parser = argparse.ArgumentParser(description="MLRL01 CLI Engine")
    parser.add_argument("--timesteps", type=int, default=10000, help="RL training steps")
    parser.add_argument("--mc_simulations", type=int, default=100, help="Monte Carlo simulations count")
    return parser.parse_args()

def main():
    args = parse_args()
    config = {
        "target_horizon": 5,
        "target_threshold": 0.005,
        "target_method": "triple_barrier",
        "train_ratio": 0.80,
        "embargo_bars": 60,
        "rl_algorithm": "RecurrentPPO",
        "rl_timesteps": args.timesteps,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "lstm_hidden": 128,
        "initial_capital": 100000.0,
        "fee_rate": 0.0001,
        "spread_cost": 0.0003,
        "slippage": 0.0002,
        "mc_simulations": args.mc_simulations,
        "data_dir": "jupiter"
    }

    print("=" * 65)
    print("  MLRL01 CLI ENTRY - USING PROGRAMMATIC ENGINE")
    print("=" * 65)

    engine = MLRL01Engine(config)
    
    # 1. Load data
    print("\n[ENGINE] Preparing data features...")
    df = engine.load_and_prepare_data()
    
    # 2. Train ML models
    print("\n[ENGINE] Training ML models...")
    results_df, predictions, close_test, dates_test = engine.train_ml_models(df)
    print("\n[ML] Model Ranking:")
    print(results_df.to_string())
    
    # 3. Train RL agent
    print("\n[ENGINE] Training RL agent...")
    rl_equity, rl_stats, rl_log, agent = engine.train_rl_agent(df)
    print(f"\n[RL] Final Equity: {rl_equity[-1]:.2f} (Trades: {rl_stats.get('total_trades', 0)})")
    
    # 4. Run Monte Carlo simulation
    print("\n[ENGINE] Running Monte Carlo simulator...")
    mc_results = engine.run_monte_carlo(rl_equity)
    print("\n[MC] Block Bootstrap results:")
    for k, v in mc_results["block_bootstrap"].items():
        print(f"  {k}: {v}")
        
    print("\n[CLI] Refactored execution complete.")

if __name__ == "__main__":
    main()