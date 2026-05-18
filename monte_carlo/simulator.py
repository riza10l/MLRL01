
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MonteCarloSimulator:
    
    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.rng = np.random.RandomState(seed)

    # ═══════════════════════════════════════════════════════════
    #  SIMULATION METHODS
    # ═══════════════════════════════════════════════════════════

    def run_block_bootstrap(self, daily_returns: np.ndarray,
                            block_size: int = 20,
                            initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) < block_size:
            return self._empty_result()

        n = len(daily_returns)
        n_blocks = max(1, n // block_size)

        all_equity = []
        all_final_returns = []
        all_max_dd = []
        all_sharpe = []

        for _ in range(self.n_sims):
            # Sample random block start indices
            starts = self.rng.randint(0, n - block_size, size=n_blocks)
            sim_returns = np.concatenate([
                daily_returns[s:s + block_size] for s in starts
            ])[:n]  # Trim to original length

            # Build equity curve
            equity = [initial_capital]
            for ret in sim_returns:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
            all_sharpe.append(self._sharpe(sim_returns))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array(all_sharpe),
            "n_sims": self.n_sims,
            "method": "block_bootstrap",
        }

    def run_regime_aware(self, daily_returns: np.ndarray,
                         volatility: np.ndarray = None,
                         initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) < 40:
            return self._empty_result()

        n = len(daily_returns)

        # Compute rolling volatility if not provided
        if volatility is None:
            volatility = pd.Series(daily_returns).rolling(20, min_periods=5).std().fillna(
                pd.Series(daily_returns).std()
            ).values

        # Classify into regimes: high vol vs low vol
        vol_median = np.median(volatility)
        high_vol_mask = volatility > vol_median
        low_vol_mask = ~high_vol_mask

        high_vol_returns = daily_returns[high_vol_mask]
        low_vol_returns = daily_returns[low_vol_mask]

        if len(high_vol_returns) == 0 or len(low_vol_returns) == 0:
            return self.run_block_bootstrap(daily_returns, initial_capital=initial_capital)

        all_equity = []
        all_final_returns = []
        all_max_dd = []
        all_sharpe = []

        for _ in range(self.n_sims):
            sim_returns = np.zeros(n)
            for i in range(n):
                if high_vol_mask[i]:
                    sim_returns[i] = self.rng.choice(high_vol_returns)
                else:
                    sim_returns[i] = self.rng.choice(low_vol_returns)

            # Add small noise for variation
            sim_returns += self.rng.normal(0, 0.0005, size=n)

            equity = [initial_capital]
            for ret in sim_returns:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
            all_sharpe.append(self._sharpe(sim_returns))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array(all_sharpe),
            "n_sims": self.n_sims,
            "method": "regime_aware",
        }

    def run_trade_resample(self, trade_returns: np.ndarray,
                           n_trades_per_sim: int = None,
                           initial_capital: float = 100_000) -> dict:
        
        if len(trade_returns) == 0:
            return self._empty_result()

        n_trades = n_trades_per_sim or len(trade_returns)

        all_equity = []
        all_final_returns = []
        all_max_dd = []
        all_sharpe = []

        for _ in range(self.n_sims):
            sampled = self.rng.choice(trade_returns, size=n_trades, replace=True)
            equity = [initial_capital]
            for ret in sampled:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))
            all_sharpe.append(self._sharpe(sampled))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array(all_sharpe),
            "n_sims": self.n_sims,
            "method": "trade_resample",
        }

    def run_return_perturbation(self, daily_returns: np.ndarray,
                                noise_std: float = 0.001,
                                initial_capital: float = 100_000) -> dict:
        
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
            "equity_curves": all_equity,
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
        
        if len(daily_returns) == 0:
            return self._empty_result()

        all_equity = []
        all_final_returns = []
        all_max_dd = []

        for _ in range(self.n_sims):
            equity = [initial_capital]
            for ret in daily_returns:
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
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array([]),
            "n_sims": self.n_sims,
            "method": "cost_variation",
        }

    def run_stress_test(self, daily_returns: np.ndarray,
                        crash_magnitude: float = -0.05,
                        crash_probability: float = 0.01,
                        spread_explosion_mult: float = 5.0,
                        initial_capital: float = 100_000) -> dict:
        
        if len(daily_returns) == 0:
            return self._empty_result()

        n = len(daily_returns)
        all_equity = []
        all_final_returns = []
        all_max_dd = []

        for _ in range(self.n_sims):
            equity = [initial_capital]
            for i, ret in enumerate(daily_returns):
                adjusted_ret = ret

                # Random flash crash
                if self.rng.random() < crash_probability:
                    adjusted_ret += crash_magnitude

                # Random spread explosion
                if self.rng.random() < crash_probability * 2:
                    adjusted_ret -= 0.001 * spread_explosion_mult

                equity.append(equity[-1] * (1 + adjusted_ret))
            equity = np.array(equity)

            all_equity.append(equity)
            all_final_returns.append((equity[-1] - initial_capital) / initial_capital)
            all_max_dd.append(self._max_drawdown(equity))

        return {
            "equity_curves": all_equity,
            "returns": np.array(all_final_returns),
            "max_drawdowns": np.array(all_max_dd),
            "sharpe_ratios": np.array([]),
            "n_sims": self.n_sims,
            "method": "stress_test",
        }

    # ═══════════════════════════════════════════════════════════
    #  REPORTING
    # ═══════════════════════════════════════════════════════════

    def generate_report(self, result: dict) -> dict:
        
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
            "dd_95th_pct": np.percentile(drawdowns, 5),
        }

        if len(sharpes) > 0:
            report["mean_sharpe"] = np.mean(sharpes)
            report["median_sharpe"] = np.median(sharpes)
            report["sharpe_5th_pct"] = np.percentile(sharpes, 5)
            report["sharpe_95th_pct"] = np.percentile(sharpes, 95)

        return report

    def plot_all(self, result: dict, save_dir: str = "results/monte_carlo"):
        
        os.makedirs(save_dir, exist_ok=True)

        self._plot_equity_fan(result, save_dir)
        self._plot_return_histogram(result, save_dir)
        self._plot_drawdown_histogram(result, save_dir)
        if len(result.get("sharpe_ratios", [])) > 0:
            self._plot_sharpe_histogram(result, save_dir)

        print(f"[MC] All plots saved to {save_dir}")

    # ═══════════════════════════════════════════════════════════
    #  PLOT FUNCTIONS
    # ═══════════════════════════════════════════════════════════

    def _plot_equity_fan(self, result: dict, save_dir: str):
        
        curves = result["equity_curves"]
        if isinstance(curves, list):
            max_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:max_len] for c in curves])
        else:
            curves_arr = curves
            max_len = curves_arr.shape[1]

        x = np.arange(max_len)
        p5 = np.percentile(curves_arr, 5, axis=0)
        p25 = np.percentile(curves_arr, 25, axis=0)
        p50 = np.percentile(curves_arr, 50, axis=0)
        p75 = np.percentile(curves_arr, 75, axis=0)
        p95 = np.percentile(curves_arr, 95, axis=0)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.fill_between(x, p5, p95, alpha=0.15, color='steelblue', label='5-95th pct')
        ax.fill_between(x, p25, p75, alpha=0.3, color='steelblue', label='25-75th pct')
        ax.plot(x, p50, color='navy', lw=2, label='Median')

        # Find worst and best paths
        final_values = curves_arr[:, -1]
        worst_idx = np.argmin(final_values)
        best_idx = np.argmax(final_values)
        ax.plot(x, curves_arr[worst_idx], color='red', lw=0.8, alpha=0.5, label='Worst path')
        ax.plot(x, curves_arr[best_idx], color='green', lw=0.8, alpha=0.5, label='Best path')

        ax.axhline(curves_arr[0, 0], color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_title(f"Monte Carlo Equity Fan ({result['n_sims']} sims — {result['method']})",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "mc_equity_fan.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_return_histogram(self, result: dict, save_dir: str):
        
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

    # ═══════════════════════════════════════════════════════════
    #  STATIC HELPERS
    # ═══════════════════════════════════════════════════════════

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
            "equity_curves": [np.array([100_000])],
            "returns": np.array([0.0]),
            "max_drawdowns": np.array([0.0]),
            "sharpe_ratios": np.array([0.0]),
            "n_sims": 0,
            "method": "empty",
        }