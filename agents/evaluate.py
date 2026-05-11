"""
Evaluation Script
==================
Evaluate trained agents, generate charts, save reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_equity_curves(all_results, rl_equity=None, benchmark_results=None,
                       dates=None, capital=100_000, save_dir="../results/plots"):
    """Equity curve semua strategy + benchmarks."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(16, 7))

    # ML models
    for name, res in all_results.items():
        eq = res.get("equity", None)
        if eq is not None:
            x = np.arange(len(eq))
            ax.plot(x, eq, lw=1.0, alpha=0.7, label=name)

    # RL
    if rl_equity is not None:
        ax.plot(np.arange(len(rl_equity)), rl_equity, lw=2.5, ls="--",
                color="black", label="PPO (RL)")

    # Benchmarks
    if benchmark_results:
        colors = {"Buy & Hold": "#e74c3c", "SMA Crossover": "#3498db", "Random": "#95a5a6"}
        for name, res in benchmark_results.items():
            eq = res.get("equity", None)
            if eq is not None:
                ax.plot(np.arange(len(eq)), eq, lw=2, ls=":",
                        color=colors.get(name, "gray"), label=f"BM: {name}")

    ax.axhline(capital, color="gray", ls=":", lw=1, alpha=0.5, label=f"Capital ({capital:,})")
    ax.set_title("Equity Curve - All Strategies vs Benchmarks", fontsize=14)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Trading Days")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()
    path = os.path.join(save_dir, "equity_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_prediction_charts(predictions, dates_test, close_test, save_dir="../results/plots"):
    """Per-model prediction vs actual."""
    ensure_dir(save_dir)
    for name, preds in predictions.items():
        safe_name = name.lower().replace(" ", "_")
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(dates_test.values, close_test.values, color="steelblue", lw=1.5, label="Close")
        axes[0].set_title(f"{name} - Test Period", fontsize=13)
        axes[0].set_ylabel("Price (USD)")
        colors = ["#e74c3c" if p == 0 else "#2ecc71" for p in preds]
        axes[1].bar(dates_test.values, preds, color=colors, width=1.5)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["DOWN", "UP"])
        axes[1].set_ylabel("Signal")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{safe_name}_prediction.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {path}")


def plot_confusion_matrices(predictions, y_test, save_dir="../results/plots"):
    """Grid confusion matrix."""
    ensure_dir(save_dir)
    n = len(predictions)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    for i, (name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(name, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Confusion Matrices", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_accuracy_comparison(results_df, save_dir="../results/plots"):
    """Bar chart accuracy."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))
    bars = ax.barh(results_df["Model"], results_df["Accuracy"], color=colors)
    ax.set_xlim(0.3, max(results_df["Accuracy"]) + 0.1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_walk_forward(wf_results, save_dir="../results/plots"):
    """Walk-forward validation results chart."""
    if wf_results.empty:
        return
    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = wf_results["test_period"]
    axes[0].bar(x, wf_results["total_return"] * 100, color="#2ecc71")
    axes[0].set_title("Return per Fold (%)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(x, wf_results["sharpe_ratio"], color="#3498db")
    axes[1].set_title("Sharpe Ratio per Fold")
    axes[1].axhline(0, color="red", ls="--", lw=1)
    axes[1].tick_params(axis="x", rotation=45)
    axes[2].bar(x, wf_results["max_drawdown"] * 100, color="#e74c3c")
    axes[2].set_title("Max Drawdown per Fold (%)")
    axes[2].tick_params(axis="x", rotation=45)
    plt.suptitle("Walk-Forward Validation Results", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "walk_forward.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def save_risk_summary(all_metrics, save_dir="../results/reports"):
    """Save risk summary CSV."""
    ensure_dir(save_dir)
    from backtest.metrics import BacktestMetrics
    table = BacktestMetrics.compare_table(all_metrics)
    path = os.path.join(save_dir, "risk_summary.csv")
    table.to_csv(path, index=False)
    print(f"\n[RISK SUMMARY]")
    print(table.to_string(index=False))
    print(f"  Saved -> {path}")
    return table


def save_comparison_csv(results_df, backtest_metrics, save_dir="../results/reports"):
    """Merge ML accuracy + backtest metrics."""
    ensure_dir(save_dir)
    merged = results_df.copy()
    bt_rows = []
    for name, m in backtest_metrics.items():
        bt_rows.append({
            "Model": name,
            "BT_Return": m.get("total_return", 0),
            "BT_MaxDD": m.get("max_drawdown", 0),
            "BT_Sharpe": m.get("sharpe_ratio", 0),
            "BT_Sortino": m.get("sortino_ratio", 0),
        })
    bt_df = pd.DataFrame(bt_rows)
    merged = merged.merge(bt_df, on="Model", how="left")
    path = os.path.join(save_dir, "comparison_results.csv")
    merged.to_csv(path, index=False)
    print(f"  Saved -> {path}")
