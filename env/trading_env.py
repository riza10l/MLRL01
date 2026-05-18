import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any

class DifferentialSharpeReward:
    def __init__(self, eta=0.02, lambda_cost=5.0, lambda_dd=2.0,
                 lambda_overtrade=0.5):
        self.eta = eta
        self.lambda_cost = lambda_cost
        self.lambda_dd = lambda_dd
        self.lambda_overtrade = lambda_overtrade
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self.peak_equity = 1.0
        self.n_trades = 0
        self.n_steps = 0

    def compute(self, step_return, friction_paid, equity_ratio, trade_executed):
        self.n_steps += 1

        # 1. Update running statistics
        self.A += self.eta * (step_return - self.A)
        self.B += self.eta * (step_return**2 - self.B)

        # 2. Differential Sharpe (numerically stable)
        denom = self.B - self.A**2
        if denom > 1e-10:
            dsr = (self.B * step_return - 0.5 * self.A * step_return**2) / (denom ** 1.5)
        else:
            dsr = step_return * 10

        # 3. Transaction cost penalty
        cost_penalty = self.lambda_cost * friction_paid

        # 4. Drawdown penalty (quadratic after 2% threshold)
        self.peak_equity = max(self.peak_equity, equity_ratio)
        dd = (self.peak_equity - equity_ratio) / (self.peak_equity + 1e-8)
        dd_penalty = self.lambda_dd * max(0, dd - 0.02) ** 2

        # 5. Overtrading penalty
        overtrade_penalty = 0.0
        if trade_executed:
            self.n_trades += 1
            trade_rate = self.n_trades / max(self.n_steps, 1)
            if trade_rate > 0.15:  # More than 15% of bars = overtrading
                overtrade_penalty = self.lambda_overtrade * (trade_rate - 0.15)

        reward = dsr - cost_penalty - dd_penalty - overtrade_penalty
        return np.clip(reward, -1, 1)

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self.peak_equity = 1.0
        self.n_trades = 0
        self.n_steps = 0

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        window_size: int = 30,
        initial_capital: float = 100_000,
        # Friction & Costs
        fee_rate: float = 0.0001,       # 0.01%
        spread_cost: float = 0.0003,    # 0.03%
        slippage: float = 0.0002,       # 0.02%
        # Constraints
        min_hold_period: int = 5,
        cooldown_after_loss: int = 3,
        max_drawdown_kill: float = 0.20,
        # DSR Reward Params
        dsr_eta: float = 0.02,
        dsr_lambda_cost: float = 5.0,
        dsr_lambda_dd: float = 2.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        # Auto-adjust window if dataset is too small
        max_window = max(1, len(self.df) - 5)
        self.window_size = min(window_size, max_window)
        self.initial_capital = initial_capital

        # Friction
        self.fee_rate = fee_rate
        self.spread_cost = spread_cost
        self.slippage = slippage
        self.total_friction = fee_rate + spread_cost + slippage

        # Constraints
        self.min_hold_period = min_hold_period
        self.cooldown_after_loss = cooldown_after_loss
        self.max_drawdown_kill = max_drawdown_kill

        # DSR Reward
        self.reward_fn = DifferentialSharpeReward(
            eta=dsr_eta, lambda_cost=dsr_lambda_cost, lambda_dd=dsr_lambda_dd
        )

        # Action Space: 0=Flat, 1=Long 50%, 2=Long 100%, 3=Short 50%, 4=Short 100%
        self.position_levels = [0.0, 0.5, 1.0, -0.5, -1.0]
        self.action_space = spaces.Discrete(len(self.position_levels))

        # Observation Space: features + portfolio state (4 dims)
        n_features = len(self.feature_columns) if self.feature_columns else 1
        obs_dim = n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.position_pct = 0.0
        self.entry_price = 0.0
        self.peak_equity = self.initial_capital
        self.steps_since_entry = 0
        self.cooldown_remaining = 0
        self.last_trade_pnl = 0.0
        self.killed = False

        self.equity_history = [self.initial_capital]
        self.returns_history = []
        self.trade_log = []

        self.reward_fn.reset()

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        price_now = self.df.iloc[self.current_step]['close']

        # Kill switch check
        if self.killed:
            self.current_step += 1
            terminated = self.current_step >= len(self.df) - 1
            self.equity_history.append(self.equity_history[-1])
            return self._get_obs(), -0.01, terminated, False, {"killed": True}

        # 1. Action Mapping & Multi-Layer Overtrading Control
        target_position = self.position_levels[action]
        trade_executed = False
        friction_paid = 0.0

        if self.position_pct != target_position:
            hold_ok = (self.position_pct == 0) or (self.steps_since_entry >= self.min_hold_period)
            cooldown_ok = self.cooldown_remaining <= 0

            if hold_ok and cooldown_ok:
                # Record PnL of closing trade
                if self.position_pct != 0 and self.entry_price > 0:
                    self.last_trade_pnl = self.position_pct * (price_now - self.entry_price) / self.entry_price

                # Execute trade with friction
                diff = abs(target_position - self.position_pct)
                friction_paid = diff * self.total_friction
                self.capital *= (1 - friction_paid)

                self.position_pct = target_position
                self.entry_price = price_now if target_position != 0 else 0.0
                self.steps_since_entry = 0
                trade_executed = True

                self.trade_log.append({
                    "step": self.current_step,
                    "price": price_now,
                    "action": target_position,
                    "cost": friction_paid * self.capital,
                    "pnl": self.last_trade_pnl,
                })

                if self.last_trade_pnl < 0:
                    self.cooldown_remaining = self.cooldown_after_loss

        # Decrement cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # 2. Update Portfolio & Equity
        unrealized_pnl = 0.0
        if self.position_pct != 0 and self.entry_price > 0:
            unrealized_pnl = (price_now - self.entry_price) / (self.entry_price + 1e-8)
            self.steps_since_entry += 1

        current_equity = self.capital * (1 + (unrealized_pnl * self.position_pct))
        self.equity_history.append(current_equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        # 3. Kill switch: stop trading if drawdown exceeds limit
        current_dd = (self.peak_equity - current_equity) / (self.peak_equity + 1e-8)
        if current_dd >= self.max_drawdown_kill:
            self.killed = True

        # 4. Calculate Reward (Differential Sharpe Ratio)
        step_return = (current_equity - self.equity_history[-2]) / (self.equity_history[-2] + 1e-8)
        self.returns_history.append(step_return)

        equity_ratio = current_equity / self.initial_capital
        reward = self.reward_fn.compute(step_return, friction_paid, equity_ratio, trade_executed)

        # 5. Step Management
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obs()

        info = {
            "equity": current_equity,
            "drawdown": current_dd,
            "position": self.position_pct,
            "step_return": step_return,
            "killed": self.killed,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Use previous step's features (the last COMPLETED bar)
        obs_step = max(0, self.current_step - 1)

        if self.feature_columns:
            latest = self.df.iloc[obs_step][self.feature_columns].values.astype(np.float32)
            latest = np.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            latest = np.zeros(1, dtype=np.float32)

        # Portfolio state
        unrealized = 0.0
        if self.entry_price > 0:
            price_prev = self.df.iloc[obs_step]['close']
            unrealized = (price_prev - self.entry_price) / (self.entry_price + 1e-8)

        dd = (self.peak_equity - self.equity_history[-1]) / (self.peak_equity + 1e-8)

        portfolio_state = np.array([
            self.position_pct,
            unrealized,
            dd,
            self.steps_since_entry / 20.0,
        ], dtype=np.float32)

        obs = np.concatenate([latest, portfolio_state])
        return obs.astype(np.float32)

    def render(self):
        print(f"Step: {self.current_step} | Equity: {self.equity_history[-1]:.2f} | Pos: {self.position_pct}")

    def get_equity_curve(self) -> np.ndarray:
        return np.array(self.equity_history)

    def get_trade_stats(self) -> Dict:
        return {
            "total_trades": len(self.trade_log),
            "total_costs": sum([t.get("cost", 0) for t in self.trade_log]),
            "killed": self.killed,
        }

    def get_trade_log(self) -> list:
        return self.trade_log