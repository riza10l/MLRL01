"""
Risk Manager V4 — Professional Risk Engine
============================================
Position sizing, drawdown tracking, kill switches,
and all risk metrics.

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd


class RiskManager:
    """
    Professional risk management with:
    - ATR-based position sizing
    - Max daily loss tracking
    - Kill switch on drawdown
    - Consecutive loss tracking
    - Exposure limits
    """

    def __init__(self, capital=100_000, risk_pct=0.02,
                 stop_loss_pct=0.01, max_drawdown=0.20,
                 max_daily_loss_pct=0.03,
                 max_consecutive_losses=5,
                 max_exposure=1.0):
        self.initial_capital = capital
        self.capital = capital
        self.risk_pct = risk_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_limit = max_drawdown
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_exposure = max_exposure

        self.peak_capital = capital
        self.equity_history = [capital]
        self.daily_start_capital = capital
        self.consecutive_losses = 0
        self.killed = False
        self.kill_reason = ""

    def position_size(self, stop_loss_amount):
        """Berapa unit yang bisa dibeli berdasarkan risiko per trade."""
        risk_amount = self.capital * self.risk_pct
        if stop_loss_amount <= 0:
            return 0
        return risk_amount / stop_loss_amount

    def position_size_atr(self, atr, contract_size=1.0):
        """
        ATR-based position sizing.
        Risk = risk_pct * capital
        Size = Risk / (ATR * contract_size)
        """
        if atr <= 0:
            return 0
        risk_amount = self.capital * self.risk_pct
        size = risk_amount / (atr * contract_size)
        return min(size, self.max_exposure * self.capital)

    def kelly_fraction(self, win_rate, avg_win, avg_loss):
        """
        Kelly criterion for optimal bet sizing.
        f* = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = 1-p
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        b = abs(avg_win / avg_loss)
        q = 1 - win_rate
        f = (win_rate * b - q) / b
        # Half-Kelly for safety
        return max(0, min(f * 0.5, self.max_exposure))

    def update_equity(self, new_capital):
        """Track equity dan update peak."""
        self.capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
        self.equity_history.append(new_capital)

    def record_trade(self, pnl):
        """Record trade result and update risk state."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check kill conditions
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.killed = True
            self.kill_reason = f"Consecutive losses: {self.consecutive_losses}"

    def new_day(self):
        """Reset daily counters."""
        self.daily_start_capital = self.capital

    def check_daily_loss(self):
        """Check if daily loss limit is hit."""
        daily_pnl = (self.capital - self.daily_start_capital) / self.daily_start_capital
        if daily_pnl <= -self.max_daily_loss_pct:
            self.killed = True
            self.kill_reason = f"Daily loss limit: {daily_pnl:.2%}"
            return True
        return False

    def current_drawdown(self):
        """Current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0
        return (self.peak_capital - self.capital) / self.peak_capital

    def should_stop_trading(self):
        """Cek apakah drawdown sudah melebihi limit atau kill switch active."""
        if self.killed:
            return True
        dd = self.current_drawdown()
        if dd >= self.max_drawdown_limit:
            self.killed = True
            self.kill_reason = f"Max drawdown: {dd:.2%}"
            return True
        return False

    def get_risk_state(self):
        """Get current risk state summary."""
        return {
            "capital": self.capital,
            "peak": self.peak_capital,
            "drawdown": self.current_drawdown(),
            "consecutive_losses": self.consecutive_losses,
            "killed": self.killed,
            "kill_reason": self.kill_reason,
        }

    # ── Static Metric Functions ───────────────────────────────

    @staticmethod
    def max_drawdown(equity_curve):
        """Hitung Maximum Drawdown (%)."""
        equity = np.asarray(equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return dd.min()

    @staticmethod
    def sharpe_ratio(returns, rf=0.0, periods=252):
        returns = np.asarray(returns)
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return (returns.mean() - rf) / returns.std() * np.sqrt(periods)

    @staticmethod
    def sortino_ratio(returns, rf=0.0, periods=252):
        returns = np.asarray(returns)
        excess = returns - rf
        downside = returns[returns < rf]
        if len(downside) < 2:
            return 0.0
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        return excess.mean() / downside_std * np.sqrt(periods)

    @staticmethod
    def calmar_ratio(returns, equity_curve, periods=252):
        mdd = abs(RiskManager.max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        ann_return = np.mean(returns) * periods
        return ann_return / mdd

    @staticmethod
    def total_return(equity_curve):
        equity = np.asarray(equity_curve)
        if len(equity) < 2 or equity[0] == 0:
            return 0.0
        return (equity[-1] - equity[0]) / equity[0]

    @staticmethod
    def annualized_return(equity_curve, periods=252):
        total_ret = RiskManager.total_return(equity_curve)
        n_periods = len(equity_curve) - 1
        if n_periods <= 0:
            return 0.0
        years = n_periods / periods
        if years <= 0:
            return 0.0
        return (1 + total_ret) ** (1 / years) - 1

    @staticmethod
    def win_rate(trades_pnl):
        if not trades_pnl:
            return 0.0
        wins = sum(1 for p in trades_pnl if p > 0)
        return wins / len(trades_pnl)

    @staticmethod
    def profit_factor(trades_pnl):
        gross_profit = sum(p for p in trades_pnl if p > 0)
        gross_loss = abs(sum(p for p in trades_pnl if p < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def tail_ratio(returns, alpha=0.05):
        """Ratio of right tail to left tail at given percentile."""
        returns = np.asarray(returns)
        if len(returns) < 20:
            return 0.0
        right = np.percentile(returns, 100 * (1 - alpha))
        left = abs(np.percentile(returns, 100 * alpha))
        if left == 0:
            return 0.0
        return right / left

    @staticmethod
    def compute_all_metrics(equity_curve, trades_pnl=None, rf=0.0, periods=252):
        equity = np.asarray(equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-10)

        metrics = {
            "total_return": RiskManager.total_return(equity),
            "annualized_return": RiskManager.annualized_return(equity, periods),
            "max_drawdown": RiskManager.max_drawdown(equity),
            "sharpe_ratio": RiskManager.sharpe_ratio(returns, rf, periods),
            "sortino_ratio": RiskManager.sortino_ratio(returns, rf, periods),
            "calmar_ratio": RiskManager.calmar_ratio(returns, equity, periods),
            "volatility": returns.std() * np.sqrt(periods) if len(returns) > 1 else 0,
            "tail_ratio": RiskManager.tail_ratio(returns),
            "n_periods": len(equity) - 1,
        }

        if trades_pnl:
            metrics["n_trades"] = len(trades_pnl)
            metrics["win_rate"] = RiskManager.win_rate(trades_pnl)
            metrics["profit_factor"] = RiskManager.profit_factor(trades_pnl)
            metrics["avg_trade_pnl"] = np.mean(trades_pnl) if trades_pnl else 0

        return metrics
