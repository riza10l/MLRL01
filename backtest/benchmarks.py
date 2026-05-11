"""
Benchmarks - Prioritas 3
==========================
Buy & Hold, SMA Crossover, Random Strategy.
RL HARUS compare sama ini.
"""

import numpy as np
import pandas as pd
from risk.risk_manager import RiskManager
from backtest.metrics import BacktestMetrics


class BenchmarkRunner:
    def __init__(self, df, initial_capital=100_000,
                 fee=0.001, spread=0.0003, slippage=0.0002):
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.total_cost_rate = fee + spread + slippage

    def run_all(self):
        results = {}
        print("\n[BENCHMARK] Running benchmarks...")
        bh_eq, bh_m = self.buy_and_hold()
        results["Buy & Hold"] = bh_m
        results["Buy & Hold"]["equity"] = bh_eq
        print(f"  Buy & Hold:    ret={bh_m['total_return']:+.2%} sharpe={bh_m['sharpe_ratio']:.3f}")

        sma_eq, sma_m = self.sma_crossover()
        results["SMA Crossover"] = sma_m
        results["SMA Crossover"]["equity"] = sma_eq
        print(f"  SMA Crossover: ret={sma_m['total_return']:+.2%} sharpe={sma_m['sharpe_ratio']:.3f}")

        rand_eq, rand_m = self.random_strategy()
        results["Random"] = rand_m
        results["Random"]["equity"] = rand_eq
        print(f"  Random:        ret={rand_m['total_return']:+.2%} sharpe={rand_m['sharpe_ratio']:.3f}")
        return results

    def buy_and_hold(self):
        closes = self.df["close"].values
        if len(closes) < 2:
            return np.array([self.initial_capital]), {"total_return": 0}
        entry_price = closes[0]
        cost = self.total_cost_rate
        cap = self.initial_capital * (1 - cost)
        equity = cap * (closes / entry_price)
        equity[-1] *= (1 - cost)
        metrics = BacktestMetrics.compute(equity, initial_capital=self.initial_capital,
                                          total_costs=cost * self.initial_capital * 2, n_trades=1)
        return equity, metrics

    def sma_crossover(self, fast=10, slow=50):
        closes = self.df["close"].values
        if len(closes) < slow + 1:
            return np.array([self.initial_capital]), {"total_return": 0}
        ma_fast = pd.Series(closes).rolling(fast).mean().values
        ma_slow = pd.Series(closes).rolling(slow).mean().values
        cap = self.initial_capital
        pos = 0
        entry = 0.0
        equity = [cap]
        trades = []
        costs = 0.0
        for i in range(slow, len(closes)):
            if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
                equity.append(cap)
                continue
            if pos == 0 and ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]:
                c = cap * self.total_cost_rate
                cap -= c
                costs += c
                pos = 1
                entry = closes[i]
            elif pos == 1 and ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]:
                pnl = (closes[i] - entry) / entry
                c = cap * self.total_cost_rate
                cap *= (1 + pnl)
                cap -= c
                costs += c
                trades.append(pnl)
                pos = 0
            if pos == 1:
                equity.append(cap * (1 + (closes[i] - entry) / entry))
            else:
                equity.append(cap)
        equity = np.array(equity)
        return equity, BacktestMetrics.compute(equity, trades, self.initial_capital, costs, len(trades))

    def random_strategy(self, seed=42, trade_prob=0.05):
        rng = np.random.RandomState(seed)
        closes = self.df["close"].values
        cap = self.initial_capital
        pos = 0
        entry = 0.0
        equity = [cap]
        trades = []
        costs = 0.0
        for i in range(1, len(closes)):
            roll = rng.random()
            if pos == 0 and roll < trade_prob:
                c = cap * self.total_cost_rate
                cap -= c
                costs += c
                pos = 1
                entry = closes[i]
            elif pos == 1 and roll < trade_prob:
                pnl = (closes[i] - entry) / entry
                c = cap * self.total_cost_rate
                cap *= (1 + pnl)
                cap -= c
                costs += c
                trades.append(pnl)
                pos = 0
            if pos == 1:
                equity.append(cap * (1 + (closes[i] - entry) / entry))
            else:
                equity.append(cap)
        equity = np.array(equity)
        return equity, BacktestMetrics.compute(equity, trades, self.initial_capital, costs, len(trades))
