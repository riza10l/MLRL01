import os
import sys
import json
import numpy as np
import pandas as pd
import redis
from typing import Dict, Tuple, Any, List

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from agents.train import (
    load_latest_data, split_data, train_ml_models as train_ml_models_fn
)
from features.features import build_production_features, get_production_feature_columns
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from risk.risk_manager import RiskManager
from monte_carlo.simulator import MonteCarloSimulator
from stable_baselines3.common.callbacks import BaseCallback

class RedisLogCallback(BaseCallback):
    """Sends training progress to Redis channel for WebSocket broadcasting."""
    def __init__(self, redis_url: str, job_id: str, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        try:
            self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            print(f"[CALLBACK ERROR] Failed to connect to Redis: {e}")
            self.redis_client = None
        self.job_id = job_id
        self.total_timesteps = total_timesteps
        
    def _on_step(self) -> bool:
        if self.redis_client is None:
            return True
            
        # Publish log every 100 steps
        if self.num_timesteps % 100 == 0 or self.num_timesteps == self.total_timesteps:
            progress_pct = min(100.0, (self.num_timesteps / self.total_timesteps) * 100.0)
            
            # Access rollout/train metrics from logger
            ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
            loss = self.logger.name_to_value.get("train/loss", 0.0)
            
            log_payload = {
                "job_id": self.job_id,
                "progress_pct": progress_pct,
                "timesteps": self.num_timesteps,
                "reward": float(ep_rew_mean),
                "loss": float(loss),
                "message": f"Step {self.num_timesteps}/{self.total_timesteps} - Progress: {progress_pct:.1f}% - Mean Reward: {ep_rew_mean:.4f} - Loss: {loss:.4f}"
            }
            
            try:
                # Publish log
                self.redis_client.publish(f"training_logs:{self.job_id}", json.dumps(log_payload))
                
                # Also save to a database-compatible logging channel / list if needed
                # We can store it as a list in Redis so FastAPI can read historical logs
                self.redis_client.rpush(f"training_history_logs:{self.job_id}", json.dumps(log_payload))
            except Exception as e:
                print(f"[CALLBACK ERROR] Redis write failed: {e}")
                
        return True

class MLRL01Engine:
    """Clean programmatic interface for MLRL01 operations."""
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config.get("data_dir", "jupiter")
        if not os.path.isabs(self.data_dir) and not os.path.exists(self.data_dir):
            # Try to resolve relative path
            self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), self.data_dir))
            
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Loads and prepares Gold Futures price data with production features."""
        df_raw = load_latest_data(self.data_dir)
        
        # Scanner for anomaly
        clean_mask = BacktestEngine.detect_anomalous_candles(df_raw)
        n_removed = (~clean_mask).sum()
        if n_removed > 0:
            df_raw = df_raw[clean_mask].reset_index(drop=True)
            print(f"[ENGINE] Anomaly cleaner removed {n_removed} rows")
            
        # Target/feature configuration
        target_horizon = self.config.get("target_horizon", 5)
        target_method = self.config.get("target_method", "triple_barrier")
        target_threshold = self.config.get("target_threshold", 0.005)
        
        # Build features
        df = build_production_features(
            df_raw,
            target_horizon=target_horizon,
            target_threshold=target_threshold,
            target_method=target_method
        )
        return df

    def train_ml_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Trains ML models on prepared dataset and evaluates accuracy/metrics."""
        feature_cols = get_production_feature_columns(df)
        train_ratio = self.config.get("train_ratio", 0.80)
        embargo = self.config.get("embargo_bars", 60)
        
        (X_train, X_test, y_train, y_test,
         X_train_sc, X_test_sc,
         dates_test, close_test, scaler) = split_data(df, feature_cols, train_ratio, embargo)
         
        predictions, results_df, trained_models = train_ml_models_fn(
            X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
        )
        
        # Return summary dataframe and raw predictions
        return results_df, predictions, close_test.values, dates_test.values

    def train_rl_agent(self, df: pd.DataFrame, job_id: str = None, redis_url: str = None) -> Tuple[np.ndarray, Dict[str, Any], List[dict], PPOAgent]:
        """Trains a PPO agent with a WebSocket-friendly progress callback."""
        feature_cols = get_production_feature_columns(df)
        train_ratio = self.config.get("train_ratio", 0.80)
        embargo = self.config.get("embargo_bars", 60)
        
        split_idx = int(len(df) * train_ratio)
        df_train_slice = df.iloc[:split_idx].reset_index(drop=True)
        test_start = split_idx + embargo
        df_test_slice = df.iloc[test_start:].reset_index(drop=True)
        
        # Create envs
        env_train = TradingEnv(
            df_train_slice, 
            feature_columns=feature_cols,
            initial_capital=self.config.get("initial_capital", 100_000.0),
            fee_rate=self.config.get("fee_rate", 0.0001),
            spread_cost=self.config.get("spread_cost", 0.0003),
            slippage=self.config.get("slippage", 0.0002)
        )
        
        rl_algorithm = self.config.get("rl_algorithm", "RecurrentPPO")
        use_lstm = (rl_algorithm == "RecurrentPPO")
        
        agent = PPOAgent(
            env_train,
            use_lstm=use_lstm,
            learning_rate=self.config.get("learning_rate", 0.0003),
            batch_size=self.config.get("batch_size", 64),
            gamma=self.config.get("gamma", 0.99),
            gae_lambda=self.config.get("gae_lambda", 0.95),
            clip_range=self.config.get("clip_range", 0.2),
            ent_coef=self.config.get("ent_coef", 0.01)
        )
        
        # Callback setup
        callback = None
        timesteps = self.config.get("rl_timesteps", 100_000)
        if job_id and redis_url:
            callback = RedisLogCallback(redis_url, job_id, timesteps)
            
        # Train
        agent.train(total_timesteps=timesteps, callback=callback)
        
        # Evaluate
        env_test = TradingEnv(
            df_test_slice, 
            feature_columns=feature_cols,
            initial_capital=self.config.get("initial_capital", 100_000.0),
            fee_rate=self.config.get("fee_rate", 0.0001),
            spread_cost=self.config.get("spread_cost", 0.0003),
            slippage=self.config.get("slippage", 0.0002)
        )
        
        rl_equity, rl_stats, rl_log = agent.evaluate(env_test)
        return rl_equity, rl_stats, rl_log, agent

    def run_backtest(self, predictions: Dict[str, np.ndarray], close_test: np.ndarray, dates_test: np.ndarray) -> Dict[str, Any]:
        """Runs the backtest engine on ML model predictions."""
        bt_engine = BacktestEngine()
        bt_results = bt_engine.run_ml_backtest(predictions, pd.Series(close_test), pd.Series(dates_test))
        
        formatted_results = {}
        for name, res in bt_results.items():
            # Calculate full suite of metrics
            metrics = RiskManager.compute_all_metrics(res)
            formatted_results[name] = {
                "equity_curve": res.tolist(),
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"],
                "volatility": metrics["volatility"]
            }
        return formatted_results

    def run_monte_carlo(self, equity_curve: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Runs Monte Carlo simulations on the strategy's returns."""
        # Calculate daily returns
        returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
        
        n_simulations = self.config.get("mc_simulations", 1000)
        mc = MonteCarloSimulator(n_simulations=n_simulations)
        
        # Run three modes
        mc_block = mc.run_block_bootstrap(returns, block_size=20)
        mc_stress = mc.run_stress_test(returns)
        mc_perturb = mc.run_return_perturbation(returns, noise_std=0.001)
        
        return {
            "block_bootstrap": mc.generate_report(mc_block),
            "stress_test": mc.generate_report(mc_stress),
            "perturbation": mc.generate_report(mc_perturb)
        }

    def run_walk_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the walk-forward validation pipeline."""
        feature_cols = get_production_feature_columns(df)
        embargo = self.config.get("embargo_bars", 60)
        
        from agents.train import walk_forward_validation
        summary = walk_forward_validation(
            df, feature_cols,
            train_years=self.config.get("wf_train_years", 3),
            test_years=self.config.get("wf_test_years", 1),
            step_years=1,
            embargo=embargo
        )
        return summary
