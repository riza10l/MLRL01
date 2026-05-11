import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Import project modules
from env.trading_env import TradingEnv
from features.features import build_production_features, robust_normalize
from agents.train import load_latest_data
from risk.risk_manager import RiskManager

def parse_args():
    parser = argparse.ArgumentParser(description="Monster Mode: Walk-Forward RL Trainer")
    parser.add_argument("--data_dir", type=str, default="jupiter", help="Directory containing price data")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per WF fold")
    parser.add_argument("--train_years", type=int, default=3, help="Training window size in years")
    parser.add_argument("--test_years", type=int, default=1, help="Testing window size in years")
    parser.add_argument("--seq_len", type=int, default=24, help="LSTM sequence length")
    parser.add_argument("--output_dir", type=str, default="results/wf_v3", help="Output for results")
    return parser.parse_args()

class WalkForwardEngine:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load and Prepare Data
        self.raw_df = load_latest_data(args.data_dir)
        self.df = build_production_features(self.raw_df)
        
        # Identify feature columns (numeric)
        self.feature_cols = [c for c in self.df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume', 'regime_state', 'returns']]
        print(f"[WF] Engine initialized. Total Features: {len(self.feature_cols)}")

    def run(self):
        df = self.df.copy()
        df['year'] = df['date'].dt.year
        min_year = df['year'].min()
        max_year = df['year'].max()
        
        all_fold_results = []

        # Sliding Window Loop
        for start_year in range(min_year, max_year - self.args.train_years - self.args.test_years + 2):
            train_end = start_year + self.args.train_years
            test_end = train_end + self.args.test_years
            
            print(f"\n{'='*60}\n[FOLD] Train: {start_year}-{train_end-1} | Test: {train_end}-{test_end-1}\n{'='*60}")
            
            df_train = df[(df['year'] >= start_year) & (df['year'] < train_end)].reset_index(drop=True)
            df_test = df[(df['year'] >= train_end) & (df['year'] < test_end)].reset_index(drop=True)
            
            if len(df_test) < 50: 
                print(f"  Skipping fold {train_end}: Not enough test data ({len(df_test)} rows)")
                continue

            # 1. Train RecurrentPPO
            model, venv = self._train_fold(df_train)
            
            # 2. Evaluate Fold
            fold_equity, fold_metrics = self._evaluate_fold(model, venv, df_test, fold_id=f"{train_end}")
            
            # 3. Baselines
            bh_equity = (df_test['close'] / df_test['close'].iloc[0]) * 100000
            
            all_fold_results.append(fold_metrics)
            
            # Log & Plot Fold
            self._plot_fold(fold_equity, bh_equity, f"fold_{train_end}")
            
        # Global Summary
        if all_fold_results:
            summary_df = pd.DataFrame(all_fold_results)
            summary_path = os.path.join(self.args.output_dir, "wf_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\n[WF] COMPLETED. Avg Sharpe: {summary_df['sharpe_ratio'].mean():.3f}")
        else:
            print("[WF] No folds completed successfully.")

    def _train_fold(self, df_train):
        # Setup Env
        def make_env():
            return TradingEnv(df_train, feature_columns=self.feature_cols)
        
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

        # Architecture Upgrade: RecurrentPPO (LSTM)
        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
            ortho_init=True
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            gae_lambda=0.95,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None
        )

        print(f"  Training RecurrentPPO for {self.args.timesteps} steps...")
        model.learn(total_timesteps=self.args.timesteps)
        return model, env

    def _evaluate_fold(self, model, venv, df_test, fold_id):
        # Use a fresh env for evaluation but wrap it so normalization from training can be applied
        test_env_raw = TradingEnv(df_test, feature_columns=self.feature_cols)
        
        # We need to apply the same normalization learned during training
        obs = test_env_raw.reset()[0]
        
        # LSTM State management
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        while not done:
            # Manually normalize observation using training venv stats
            norm_obs = venv.normalize_obs(obs)
            
            action, lstm_states = model.predict(
                norm_obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            obs, reward, terminated, truncated, info = test_env_raw.step(action.item())
            done = terminated or truncated
            episode_starts = np.zeros((1,), dtype=bool)

        equity = test_env_raw.equity_history
        metrics = RiskManager.compute_all_metrics(np.array(equity))
        metrics['fold'] = fold_id
        return equity, metrics

    def _plot_fold(self, rl_equity, bh_equity, name):
        plt.figure(figsize=(12, 6))
        plt.plot(rl_equity, label='RecurrentPPO (Monster)', color='royalblue', lw=2)
        plt.plot(bh_equity.values, label='Buy & Hold', color='gray', linestyle='--')
        plt.title(f"Equity Curve - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.args.output_dir, f"{name}.png"))
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    engine = WalkForwardEngine(args)
    engine.run()
