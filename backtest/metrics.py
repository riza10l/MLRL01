"""
Backtest Metrics
=================
Centralized metric computation for backtesting.

"ill keep evolving till i die" ahh machine
"""

import numpy as np
import pandas as pd
from risk.risk_manager import RiskManager


class BacktestMetrics:
    """Compute and format all backtest metrics."""

    @staticmethod
    def compute(equity_curve, trades_pnl=None, initial_capital=100_000,
                total_costs=0.0, n_trades=0):
        """
        Compute full metrics from equity curve.
        
        Returns dict with all metrics.
        """
        equity = np.asarray(equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-10)

        metrics = RiskManager.compute_all_metrics(equity, trades_pnl)
        metrics["initial_capital"] = initial_capital
        metrics["final_capital"] = equity[-1]
        metrics["total_costs"] = total_costs
        metrics["n_trades"] = n_trades if not trades_pnl else len(trades_pnl)

        return metrics

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        """Pretty print metrics."""
        lines = []
        lines.append(f"  Return:     {metrics.get('total_return', 0):+.2%}")
        lines.append(f"  Ann Return: {metrics.get('annualized_return', 0):+.2%}")
        lines.append(f"  Max DD:     {metrics.get('max_drawdown', 0):.2%}")
        lines.append(f"  Sharpe:     {metrics.get('sharpe_ratio', 0):.3f}")
        lines.append(f"  Sortino:    {metrics.get('sortino_ratio', 0):.3f}")
        lines.append(f"  Calmar:     {metrics.get('calmar_ratio', 0):.3f}")
        lines.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        lines.append(f"  Trades:     {metrics.get('n_trades', 0)}")
        if "win_rate" in metrics:
            lines.append(f"  Win Rate:   {metrics.get('win_rate', 0):.1%}")
        if "profit_factor" in metrics:
            lines.append(f"  Profit F:   {metrics.get('profit_factor', 0):.2f}")
        lines.append(f"  Costs:      ${metrics.get('total_costs', 0):,.2f}")
        return "\n".join(lines)

    @staticmethod
    def compare_table(results: dict) -> pd.DataFrame:
        """
        Create comparison DataFrame from dict of {name: metrics}.
        """
        rows = []
        for name, m in results.items():
            rows.append({
                "Strategy": name,
                "Return (%)": f"{m.get('total_return', 0):+.2%}",
                "Max DD (%)": f"{m.get('max_drawdown', 0):.2%}",
                "Sharpe": f"{m.get('sharpe_ratio', 0):.3f}",
                "Sortino": f"{m.get('sortino_ratio', 0):.3f}",
                "Calmar": f"{m.get('calmar_ratio', 0):.3f}",
                "Trades": m.get("n_trades", 0),
                "Win Rate": f"{m.get('win_rate', 0):.1%}" if "win_rate" in m else "N/A",
                "Costs": f"${m.get('total_costs', 0):,.0f}",
            })
        return pd.DataFrame(rows)
