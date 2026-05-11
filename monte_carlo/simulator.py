"""
Monte Carlo Simulation Module
===============================
Institutional-grade Monte Carlo analysis for trading strategy validation.

Why Monte Carlo matters:
  - A single backtest is ONE path through history
  - Monte Carlo generates 1000+ possible paths
  - Shows probability of ruin, worst-case scenarios
  - Quantifies luck vs skill in performance
  - Required for institutional strategy approval

Methods:
  1. Trade resampling (bootstrap trades with replacement)
  2. Equity curve perturbation (add random noise to returns)
  3. Randomized costs (vary slippage/spread within realistic range)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MonteCarloSimulator:
    """
    Professional Monte Carlo simulation for trading strategy validation.
    """
    
    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.rng = np.random.RandomState(seed)
    
    def run_trade_resample(self, trade_returns: np.ndarray, 
                           n_trades_per_sim: int = None,
                           initial_capital: float = 100_000) -> dict:
        """
        Bootstrap resampling of trade returns.
        Randomly reorders trades to see how path-dependent results are.
        
        Why: If your strategy is robust, shuffling trade order shouldn't
             destroy performance. If it does, you're relying on lucky sequencing.
        """
        if len(trade_returns) == 0:
            return self._empty_result()
        
        n_trades = n_trades_per_sim or len(trade_returns)
        
        all_equity = []
        all_final_returns = []
        all_max_dd = []
        all_sharpe = []
        
        for _ in range(self.n_sims):
            # Resample trades with replacement
            sampled = self.rng.choice(trade_returns, size=n_trades, replace=True)
            
            # Build equity curve
            equity = [initial_capital]
            for ret in sampled:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)
            
            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
            all_sharpe.append(self._sharpe(sampled))
        
        return {
            "equity_curves": np.array(all_equity),
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array(all_sharpe),
            "n_sims": self.n_sims,
            "method": "trade_resample",
        }
    
    def run_return_perturbation(self, daily_returns: np.ndarray,
                                noise_std: float = 0.001,
                                initial_capital: float = 100_000) -> dict:
        """
        Add random noise to daily returns to simulate model uncertainty.
        
        Why: Your backtest returns are point estimates. In reality, 
             execution varies. This shows the range of possible outcomes.
        """
        if len(daily_returns) == 0:
            return self._empty_result()
        
        all_equity = []
        all_final_returns = []
        all_max_dd = []
        all_sharpe = []
        
        for _ in range(self.n_sims):
            noise = self.rng.normal(0, noise_std, size=len(daily_returns))
            perturbed = daily_returns + noise
            
            equity = [initial_capital]
            for ret in perturbed:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)
            
            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
            all_sharpe.append(self._sharpe(perturbed))
        
        return {
            "equity_curves": np.array(all_equity),
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array(all_sharpe),
            "n_sims": self.n_sims,
            "method": "return_perturbation",
        }
    
    def run_cost_variation(self, daily_returns: np.ndarray,
                           base_cost_per_trade: float = 0.0006,
                           cost_std: float = 0.0003,
                           avg_trades_per_day: float = 0.1,
                           initial_capital: float = 100_000) -> dict:
        """
        Vary transaction costs randomly to simulate real-world execution.
        
        Why: Slippage and spread vary intraday. Your fixed cost assumption
             may be optimistic. This shows cost sensitivity.
        """
        if len(daily_returns) == 0:
            return self._empty_result()
        
        all_equity = []
        all_final_returns = []
        all_max_dd = []
        
        for _ in range(self.n_sims):
            equity = [initial_capital]
            for ret in daily_returns:
                # Random cost per day
                trade_today = self.rng.random() < avg_trades_per_day
                if trade_today:
                    cost = max(0, self.rng.normal(base_cost_per_trade, cost_std))
                    adjusted_ret = ret - cost
                else:
                    adjusted_ret = ret
                equity.append(equity[-1] * (1 + adjusted_ret))
            equity = np.array(equity)
            
            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
        
        return {
            "equity_curves": np.array(all_equity),
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array([]),
            "n_sims": self.n_sims,
            "method": "cost_variation",
        }
    
    def generate_report(self, result: dict) -> dict:
        """Generate statistical summary from MC results."""
        returns = result["returns"]
        drawdowns = result["max_drawdowns"]
        sharpes = result.get("sharpe_ratios", np.array([]))
        
        report = {
            "method": result["method"],
            "n_simulations": result["n_sims"],
            # Returns
            "mean_return": np.mean(returns),
            "median_return": np.median(returns),
            "std_return": np.std(returns),
            "return_5th_pct": np.percentile(returns, 5),
            "return_25th_pct": np.percentile(returns, 25),
            "return_75th_pct": np.percentile(returns, 75),
            "return_95th_pct": np.percentile(returns, 95),
            # Risk
            "prob_positive": np.mean(returns > 0),
            "prob_ruin_10pct": np.mean(returns < -0.10),
            "prob_ruin_20pct": np.mean(returns < -0.20),
            "mean_max_dd": np.mean(drawdowns),
            "worst_max_dd": np.min(drawdowns),
            "dd_95th_pct": np.percentile(drawdowns, 5),  # 5th pct of DD = worst 5%
        }
        
        if len(sharpes) > 0:
            report["mean_sharpe"] = np.mean(sharpes)
            report["median_sharpe"] = np.median(sharpes)
            report["sharpe_5th_pct"] = np.percentile(sharpes, 5)
            report["sharpe_95th_pct"] = np.percentile(sharpes, 95)
        
        return report
    
    def plot_all(self, result: dict, save_dir: str = "results/monte_carlo"):
        """Generate all Monte Carlo visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        self._plot_equity_fan(result, save_dir)
        self._plot_return_histogram(result, save_dir)
        self._plot_drawdown_histogram(result, save_dir)
        if len(result.get("sharpe_ratios", [])) > 0:
            self._plot_sharpe_histogram(result, save_dir)
        
        print(f"[MC] All plots saved to {save_dir}")
    
    # ── Plot Functions ────────────────────────────────────────
    
    def _plot_equity_fan(self, result: dict, save_dir: str):
        """Fan chart showing percentile bands of equity paths."""
        curves = result["equity_curves"]
        max_len = min(c.shape[0] for c in curves) if isinstance(curves, list) else curves.shape[1]
        curves_trimmed = np.array([c[:max_len] for c in curves])
        
        x = np.arange(max_len)
        p5 = np.percentile(curves_trimmed, 5, axis=0)
        p25 = np.percentile(curves_trimmed, 25, axis=0)
        p50 = np.percentile(curves_trimmed, 50, axis=0)
        p75 = np.percentile(curves_trimmed, 75, axis=0)
        p95 = np.percentile(curves_trimmed, 95, axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.fill_between(x, p5, p95, alpha=0.15, color='steelblue', label='5-95th pct')
        ax.fill_between(x, p25, p75, alpha=0.3, color='steelblue', label='25-75th pct')
        ax.plot(x, p50, color='navy', lw=2, label='Median')
        ax.plot(x, curves_trimmed[0], color='red', lw=0.8, alpha=0.5, label='Worst path')
        ax.plot(x, curves_trimmed[-1], color='green', lw=0.8, alpha=0.5, label='Best path')
        
        ax.axhline(curves_trimmed[0, 0], color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_title(f"Monte Carlo Equity Fan ({result['n_sims']} sims)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_equity_fan.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_return_histogram(self, result: dict, save_dir: str):
        """Distribution of final returns."""
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = result["returns"] * 100
        
        ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='navy', lw=0.5)
        ax.axvline(np.mean(returns), color='red', ls='--', lw=2, label=f'Mean: {np.mean(returns):.1f}%')
        ax.axvline(np.median(returns), color='orange', ls='--', lw=2, label=f'Median: {np.median(returns):.1f}%')
        ax.axvline(0, color='black', ls='-', lw=1)
        
        prob_pos = np.mean(result["returns"] > 0) * 100
        ax.set_title(f"MC Return Distribution ({prob_pos:.0f}% positive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Total Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_return_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_drawdown_histogram(self, result: dict, save_dir: str):
        """Distribution of max drawdowns."""
        fig, ax = plt.subplots(figsize=(10, 6))
        dd = result["max_drawdowns"] * 100
        
        ax.hist(dd, bins=50, color='#e74c3c', alpha=0.7, edgecolor='darkred', lw=0.5)
        ax.axvline(np.mean(dd), color='navy', ls='--', lw=2, label=f'Mean: {np.mean(dd):.1f}%')
        ax.axvline(np.percentile(dd, 5), color='black', ls=':', lw=2, label=f'5th pct: {np.percentile(dd, 5):.1f}%')
        
        ax.set_title("MC Max Drawdown Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_drawdown_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_sharpe_histogram(self, result: dict, save_dir: str):
        """Distribution of Sharpe ratios."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sharpes = result["sharpe_ratios"]
        
        ax.hist(sharpes, bins=50, color='#2ecc71', alpha=0.7, edgecolor='darkgreen', lw=0.5)
        ax.axvline(np.mean(sharpes), color='red', ls='--', lw=2, label=f'Mean: {np.mean(sharpes):.2f}')
        ax.axvline(0, color='black', ls='-', lw=1)
        ax.axvline(np.percentile(sharpes, 5), color='gray', ls=':', lw=2, 
                   label=f'5th pct: {np.percentile(sharpes, 5):.2f}')
        
        prob_pos = np.mean(sharpes > 0) * 100
        ax.set_title(f"MC Sharpe Distribution ({prob_pos:.0f}% positive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_sharpe_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ── Static Helpers ────────────────────────────────────────
    
    @staticmethod
    def _max_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return dd.min()
    
    @staticmethod
    def _sharpe(returns, periods=252):
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods)
    
    @staticmethod
    def _empty_result():
        return {
            "equity_curves": np.array([]),
            "returns": np.array([0.0]),
            "max_drawdowns": np.array([0.0]),
            "sharpe_ratios": np.array([0.0]),
            "n_sims": 0,
            "method": "empty",
        }
