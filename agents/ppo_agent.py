
import os
import numpy as np
import torch
from stable_baselines3 import PPO

class PPOAgent:
    
    def __init__(
        self,
        env,
        policy="MlpPolicy",
        use_lstm=False,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    ):
        self.env = env
        self.use_lstm = use_lstm
        self.verbose = verbose
        
        # Explicitly check for GPU acceleration availability
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AGENT] PyTorch detected device: {torch_device.upper()} (CUDA Available: {torch.cuda.is_available()})")

        if use_lstm:
            # Prioritas 9: Recurrent PPO
            try:
                from sb3_contrib import RecurrentPPO
                self.model = RecurrentPPO(
                    "MlpLstmPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    verbose=verbose,
                )
                print("[AGENT] Using RecurrentPPO (LSTM Policy) - Prioritas 9")
            except ImportError:
                print("[AGENT] sb3-contrib not installed, falling back to standard PPO")
                print("[AGENT] Install with: pip install sb3-contrib")
                self.use_lstm = False
                self.model = PPO(
                    policy, env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    verbose=verbose,
                )
        else:
            self.model = PPO(
                policy, env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=verbose,
            )
            print(f"[AGENT] Using PPO ({policy})")

    def train(self, total_timesteps=100_000, progress_bar=False, callback=None):
        
        print(f"[AGENT] Training PPO ({total_timesteps:,} timesteps)...")
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=callback,
        )
        print("[AGENT] Training complete!")
        return self.model

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        
        return self.model.predict(
            obs, 
            state=state, 
            episode_start=episode_start, 
            deterministic=deterministic
        )

    def evaluate(self, env, n_episodes=1):
        
        all_equity = []
        all_stats = []
        all_logs = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            lstm_states = None
            # Episode start mask for RecurrentPPO
            episode_start = np.ones((1,), dtype=bool)

            while not (terminated or truncated):
                if self.use_lstm:
                    action, lstm_states = self.model.predict(
                        obs, 
                        state=lstm_states, 
                        episode_start=episode_start,
                        deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)

            equity = env.get_equity_curve()
            stats = env.get_trade_stats()
            log = env.get_trade_log()

            all_equity.append(equity)
            all_stats.append(stats)
            all_logs.append(log)

        # Return results of the last episode for compatibility with main.py
        # but can be extended to return all
        return all_equity[-1], all_stats[-1], all_logs[-1]

    def save(self, path):
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"[AGENT] Model saved -> {path}")

    def load(self, path, env=None):
        
        if self.use_lstm:
            try:
                from sb3_contrib import RecurrentPPO
                self.model = RecurrentPPO.load(path, env=env)
            except ImportError:
                self.model = PPO.load(path, env=env)
        else:
            self.model = PPO.load(path, env=env)
        print(f"[AGENT] Model loaded <- {path}")
        return self.model