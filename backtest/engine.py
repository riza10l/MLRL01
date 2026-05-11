"""
Backtest Engine
"""

import numpy as np
import pandas as pd
from risk.risk_manager import RiskManager
from backtest.metrics import BacktestMetrics


class BacktestEngine:
    def __init__(self, initial_capital=100_000, stop_loss=0.01,
                 take_profit=0.015, fee=0.001, spread=0.0003, slippage=0.0002):
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.total_cost_rate = fee + spread + slippage

    def run_ml_backtest(self, predictions, close_test, dates_test=None):
        """Backtest semua ML model signals. Returns dict of {name: metrics}."""
        results = {}
        print("\n[BACKTEST] Running ML backtests...")
        for name, preds in predictions.items():
            equity, metrics = self._backtest_single(preds, close_test.values)
            metrics["equity"] = equity
            results[name] = metrics
            print(f"  {name:<22} ret={metrics['total_return']:+.2%} "
                  f"sharpe={metrics['sharpe_ratio']:.3f} trades={metrics['n_trades']}")
        return results

    def _backtest_single(self, preds, closes):
        cap = self.initial_capital
        pos = 0
        entry = 0.0
        equity = [cap]
        trades = []
        costs = 0.0

        for i in range(len(preds)):
            if i + 1 >= len(closes):
                break
            price = closes[i]
            price_next = closes[i + 1]

            if pos == 1:
                pnl_pct = (price - entry) / entry
                if pnl_pct <= -self.stop_loss:
                    c = cap * self.total_cost_rate
                    cap *= (1 - self.stop_loss)
                    cap -= c
                    costs += c
                    trades.append(-self.stop_loss)
                    pos = 0
                elif pnl_pct >= self.take_profit:
                    c = cap * self.total_cost_rate
                    cap *= (1 + self.take_profit)
                    cap -= c
                    costs += c
                    trades.append(self.take_profit)
                    pos = 0
                elif preds[i] == 0:
                    c = cap * self.total_cost_rate
                    cap *= (1 + pnl_pct)
                    cap -= c
                    costs += c
                    trades.append(pnl_pct)
                    pos = 0
            else:
                if preds[i] == 1:
                    c = cap * self.total_cost_rate
                    cap -= c
                    costs += c
                    pos = 1
                    entry = price_next
            equity.append(cap if pos == 0 else cap * (1 + (closes[min(i+1, len(closes)-1)] - entry) / entry))

        equity = np.array(equity)
        metrics = BacktestMetrics.compute(equity, trades, self.initial_capital, costs, len(trades))
        return equity, metrics
