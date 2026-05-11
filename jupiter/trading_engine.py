"""
TRADING ENGINE - Unified ML + RL Trading System
Gold Futures (GC=F) - Daily Strategy
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
#ml 
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, confusion_matrix,
                                     ConfusionMatrixDisplay)
from sklearn.preprocessing   import StandardScaler

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
#rl
import gymnasium as gym
from stable_baselines3 import PPO

matplotlib.rcParams.update({
    "figure.dpi"        : 120,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
})

#setting, misalkan mau dipake diubah dulu, semuanya bisa di setting sesuai lo sendiri men
SYMBOL          = "GC=F"          # Instrumen (gold futures)
TRAIN_RATIO     = 0.80            # 80% data buat training
FEATURE_COLS    = ["return", "range", "ma10", "ma50",
                   "volatility", "ma10_ma50_diff",
                   "vol_ma10", "price_ma10"]
RL_TIMESTEPS    = 10_000          # Total timesteps training PPO
RL_WINDOW       = 30              # Jumlah hari buat observasi RL

# ── Risk Management Config ─────────────────────────────────
INITIAL_CAPITAL = 10_000         # Modal awal (USD)
RISK_PER_TRADE  = 0.02            # Risiko per trade (2% dari modal)
STOP_LOSS_PCT   = 0.01           # Stop-loss 1%
TAKE_PROFIT_PCT = 0.015            # Take-profit 1.5%
MAX_DRAWDOWN    = 0.20            # Max drawdown tolerated (20%)

# load data

def load_latest_data():
    """Auto-load file dataXX.csv yang paling baru."""
    files = glob.glob("data*.csv")
    if not files:
        raise FileNotFoundError("Waduh, file data*.csv nggak ketemu! Jalanin call.py dulu.")
    latest = max(files, key=os.path.getctime)
    print(f"[DATA] Pake file terbaru: {latest}")

    df = pd.read_csv(latest)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    print(f"[DATA] Rows: {len(df):,}  |  {df['date'].min().date()} -> {df['date'].max().date()}")
    return df

# Fungsi 

def build_features(df):
    """Tambahin fitur teknikal."""
    df = df.copy()
    df["return"]         = df["close"].pct_change()
    df["range"]          = df["high"] - df["low"]
    df["ma10"]           = df["close"].rolling(10).mean()
    df["ma50"]           = df["close"].rolling(50).mean()
    df["volatility"]     = df["close"].rolling(10).std()
    df["ma10_ma50_diff"] = df["ma10"] - df["ma50"]
    df["vol_ma10"]       = df["volume"].rolling(10).mean()
    df["price_ma10"]     = df["close"] / df["ma10"] - 1

    # Target: 1 = besok harga naik, 0 = turun/sama
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[FEAT] Setelah feature engineering: {len(df):,} rows")
    return df

# train test lalu split cees
def split_data(df):
    """Split 80/20, time-series style (no shuffle!)."""
    split_idx = int(len(df) * TRAIN_RATIO)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    dates_test = df["date"].iloc[split_idx:].reset_index(drop=True)
    close_test = df["close"].iloc[split_idx:].reset_index(drop=True)

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[SPLIT] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, dates_test, close_test, scaler

# Ml models

SCALED_MODELS = {"Logistic Regression", "SVM"}

def train_ml_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    """Latih semua model ML, return predictions + metrics."""
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost"             : XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
        "LightGBM"            : LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
        "SVM"                 : SVC(kernel="rbf", probability=True, random_state=42),
    }

    predictions = {}
    results     = []

    for name, model in models.items():
        Xtr = X_train_sc if name in SCALED_MODELS else X_train
        Xte = X_test_sc  if name in SCALED_MODELS else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        predictions[name] = y_pred

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec})
        print(f"  {name:<22}  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    results_df.index += 1
    return predictions, results_df, models

# Setup RL env

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment buat RL trading.

    State  : Window RL_WINDOW hari harga close (normalized)
    Action : 0=Hold, 1=Buy, 2=Sell
    Reward : Profit/loss berdasarkan aksi + penalti drawdown
    """

    def __init__(self, df, window=RL_WINDOW,
                 initial_capital=INITIAL_CAPITAL,
                 stop_loss=STOP_LOSS_PCT,
                 take_profit=TAKE_PROFIT_PCT):
        super().__init__()
        self.df             = df.reset_index(drop=True)
        self.window         = window
        self.initial_capital= initial_capital
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit

        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space      = gym.spaces.Discrete(3)
        # Observation: window harga close (log-return)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.capital      = self.initial_capital
        self.position     = 0      # 0=flat, 1=long
        self.entry_price  = 0.0
        self.equity_curve = [self.initial_capital]
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        price_now  = self.df["close"].iloc[self.current_step]
        price_prev = self.df["close"].iloc[self.current_step - 1]
        reward     = 0.0

        if self.position == 1:  # Lagi punya posisi long
            pnl_pct = (price_now - self.entry_price) / self.entry_price

            # Stop-loss triggered
            if pnl_pct <= -self.stop_loss:
                reward          = -self.stop_loss * self.capital
                self.capital   *= (1 - self.stop_loss)
                self.position   = 0

            # Take-profit triggered
            elif pnl_pct >= self.take_profit:
                reward          = self.take_profit * self.capital
                self.capital   *= (1 + self.take_profit)
                self.position   = 0

            # Agent milih Sell -> tutup posisi
            elif action == 2:
                reward         = pnl_pct * self.capital
                self.capital  *= (1 + pnl_pct)
                self.position  = 0

        else:  # Posisi flat
            if action == 1:  # Buy -> buka posisi long
                self.position   = 1
                self.entry_price= price_now

        self.equity_curve.append(self.capital)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs  = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        """Log-return dari window terakhir sebagai observasi."""
        window_data = self.df["close"].iloc[
            self.current_step - self.window : self.current_step
        ].values.astype(np.float32)
        log_returns = np.diff(np.log(window_data + 1e-8))
        if len(log_returns) < self.window:
            log_returns = np.pad(log_returns, (self.window - len(log_returns), 0))
        return log_returns.astype(np.float32)

    def get_equity_curve(self):
        return np.array(self.equity_curve)

def train_rl_model(df):
    """Latih PPO agent, return model + equity curve."""
    print(f"\n[RL] Training PPO agent ({RL_TIMESTEPS:,} timesteps)...")
    env   = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=RL_TIMESTEPS)
    model.save("ppo_trading_gold")
    print("[RL] Model disimpan -> ppo_trading_gold")

    # Ambil equity curve dari episode terakhir
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    equity = env.get_equity_curve()
    return model, equity

# Risk management

class RiskManager:
    """Kalkulasi position sizing, max drawdown, Sharpe ratio."""

    def __init__(self, capital=INITIAL_CAPITAL, risk_pct=RISK_PER_TRADE):
        self.capital   = capital
        self.risk_pct  = risk_pct

    def position_size(self, stop_loss_amount):
        """Berapa unit yang bisa dibeli berdasarkan risiko."""
        risk_amount = self.capital * self.risk_pct
        if stop_loss_amount <= 0:
            return 0
        return risk_amount / stop_loss_amount

    @staticmethod
    def max_drawdown(equity_curve):
        """Hitung Maximum Drawdown (%)."""
        peak = np.maximum.accumulate(equity_curve)
        dd   = (equity_curve - peak) / peak
        return dd.min()

    @staticmethod
    def sharpe_ratio(returns, rf=0.0, periods=252):
        """Hitung Sharpe Ratio (annualized)."""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() - rf) / returns.std() * np.sqrt(periods)

# Backtest sinyal ml

def backtest_ml(predictions, close_test, dates_test,
                capital=INITIAL_CAPITAL,
                stop_loss=STOP_LOSS_PCT,
                take_profit=TAKE_PROFIT_PCT):
    """
    Backtest sederhana untuk setiap ML model.
    Signal 1 = Buy next day, 0 = hold/skip.
    """
    rm = RiskManager(capital)
    backtest_results = {}

    for name, preds in predictions.items():
        cap       = capital
        position  = 0
        entry     = 0.0
        equity    = [cap]
        trades    = []

        closes = close_test.values
        for i, (pred, price) in enumerate(zip(preds, closes)):
            if i + 1 >= len(closes):
                break
            price_next = closes[i + 1]

            if position == 1:
                pnl_pct = (price - entry) / entry
                # Stop-loss
                if pnl_pct <= -stop_loss:
                    cap    *= (1 - stop_loss)
                    position = 0
                    trades.append(pnl_pct)
                # Take-profit
                elif pnl_pct >= take_profit:
                    cap    *= (1 + take_profit)
                    position = 0
                    trades.append(pnl_pct)
                # Model bilang sell
                elif pred == 0:
                    cap    *= (1 + pnl_pct)
                    position = 0
                    trades.append(pnl_pct)
            else:
                if pred == 1:  # Sinyal Buy
                    position = 1
                    entry    = price_next  # beli di open hari berikutnya

            equity.append(cap)

        equity_arr = np.array(equity)
        rets       = np.diff(equity_arr) / equity_arr[:-1]
        mdd        = RiskManager.max_drawdown(equity_arr)
        sharpe     = RiskManager.sharpe_ratio(rets)
        total_ret  = (cap - capital) / capital

        backtest_results[name] = {
            "equity"    : equity_arr,
            "total_ret" : total_ret,
            "mdd"       : mdd,
            "sharpe"    : sharpe,
            "n_trades"  : len(trades),
        }
        print(f"  {name:<22}  ret={total_ret:+.2%}  mdd={mdd:.2%}  sharpe={sharpe:.2f}  trades={len(trades)}")

    return backtest_results

# output chart csv 

def plot_prediction_charts(predictions, dates_test, close_test):
    """Per-model prediction vs actual (sama kayak notebook)."""
    for name, preds in predictions.items():
        safe_name = name.lower().replace(" ", "_")
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={"height_ratios": [3, 1]})

        ax1 = axes[0]
        ax1.plot(dates_test.values, close_test.values, color="steelblue", lw=1.5, label="Close")
        ax1.set_title(f"{name}  —  Test Period Prediction vs Actual", fontsize=13)
        ax1.set_ylabel("Price (USD)")

        colors = ["#e74c3c" if p == 0 else "#2ecc71" for p in preds]
        ax2 = axes[1]
        ax2.bar(dates_test.values, preds, color=colors, width=1.5)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["DOWN", "UP"])
        ax2.set_ylabel("Signal")
        ax2.set_xlabel("Date")

        plt.tight_layout()
        fname = f"{safe_name}_prediction.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {fname}")

def plot_confusion_matrices(predictions, y_test):
    """Grid confusion matrix untuk semua model."""
    n     = len(predictions)
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

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig("confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> confusion_matrices.png")

def plot_accuracy_comparison(results_df):
    """Bar chart accuracy semua model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))
    bars    = ax.barh(results_df["Model"], results_df["Accuracy"], color=colors)
    ax.set_xlim(0.3, max(results_df["Accuracy"]) + 0.1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig("accuracy_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> accuracy_comparison.png")

def plot_backtest_equity(backtest_results, rl_equity=None, dates_test=None, capital=INITIAL_CAPITAL):
    """Equity curve semua model + RL."""
    fig, ax = plt.subplots(figsize=(16, 7))

    for name, res in backtest_results.items():
        eq = res["equity"]
        if dates_test is not None and len(eq) <= len(dates_test):
            x = dates_test.values[:len(eq)]
        else:
            x = np.arange(len(eq))
        ax.plot(x, eq, lw=1.2, label=name)

    if rl_equity is not None:
        if dates_test is not None and len(rl_equity) <= len(dates_test):
            x_rl = dates_test.values[:len(rl_equity)]
        else:
            x_rl = np.arange(len(rl_equity))
        ax.plot(x_rl, rl_equity, lw=2, ls="--", color="black", label="PPO (RL)")

    ax.axhline(capital, color="gray", ls=":", lw=1, label=f"Capital Awal ({capital:,})")
    ax.set_title("Backtest — Equity Curve Semua Model", fontsize=13)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    fig.savefig("backtest_equity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> backtest_equity.png")

def plot_risk_summary(backtest_results, rl_equity=None, capital=INITIAL_CAPITAL):
    """Heatmap/tabel: return, MDD, Sharpe."""
    rows = []
    for name, res in backtest_results.items():
        rows.append({
            "Model"     : name,
            "Return (%)" : f"{res['total_ret']:+.2%}",
            "Max DD (%)" : f"{res['mdd']:.2%}",
            "Sharpe"    : f"{res['sharpe']:.2f}",
            "Trades"    : res['n_trades'],
        })

    if rl_equity is not None:
        rl_ret    = (rl_equity[-1] - capital) / capital
        rl_rets   = np.diff(rl_equity) / rl_equity[:-1]
        rl_mdd    = RiskManager.max_drawdown(rl_equity)
        rl_sharpe = RiskManager.sharpe_ratio(rl_rets)
        rows.append({
            "Model"      : "PPO (RL)",
            "Return (%)": f"{rl_ret:+.2%}",
            "Max DD (%)": f"{rl_mdd:.2%}",
            "Sharpe"    : f"{rl_sharpe:.2f}",
            "Trades"    : "N/A",
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("risk_summary.csv", index=False)
    print("\n[RISK SUMMARY]")
    print(summary_df.to_string(index=False))
    print("  Saved -> risk_summary.csv")

def save_comparison_csv(results_df, backtest_results):
    """Gabungin ML metrics + backtest ke satu CSV."""
    merged = results_df.copy()
    bt_rows = []
    for name, res in backtest_results.items():
        bt_rows.append({
            "Model"      : name,
            "BT_Return"  : res["total_ret"],
            "BT_MaxDD"   : res["mdd"],
            "BT_Sharpe"  : res["sharpe"],
        })
    bt_df  = pd.DataFrame(bt_rows)
    merged = merged.merge(bt_df, on="Model", how="left")
    merged.to_csv("comparison_results.csv", index=False)
    print("  Saved -> comparison_results.csv")

# main program

if __name__ == "__main__":
    print("=" * 60)
    print("  TRADING ENGINE  -  Gold Futures (GC=F)")
    print("=" * 60)

    # 1. Load data
    df_raw = load_latest_data()

    # 2. Feature engineering
    df = build_features(df_raw)

    # 3. Split data
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     dates_test, close_test, scaler) = split_data(df)

    # Training model ml
    print("\n[ML] Training semua model...")
    predictions, results_df, trained_models = train_ml_models(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )
    print("\n[ML] Model Ranking (by Accuracy):")
    print(results_df.to_string())

    # Training model rl
    # Kita pecah data: belajar di data train, ngetes di data test
    df_train_slice = df.iloc[:int(len(df) * TRAIN_RATIO)].reset_index(drop=True)
    df_test_slice  = df.iloc[int(len(df) * TRAIN_RATIO):].reset_index(drop=True)
    
    # BELAJAR (Training) di data masa lalu
    print("\n[RL] Agent lagi disekolahin (belajar di data train)...")
    rl_model, _ = train_rl_model(df_train_slice) 
    
    # TEST (Evaluation) di data test buat liat performa asli
    print("[RL] Agent lagi tes ilmu (testing di data baru)...")
    env_test = TradingEnv(df_test_slice)
    obs, _ = env_test.reset()
    done = False
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_test.step(action)
    rl_equity = env_test.get_equity_curve()


    # backtest
    print("\n[BACKTEST] Menghitung equity curve semua model...")
    backtest_results = backtest_ml(predictions, close_test, dates_test)

    # chart
    print("\n[CHARTS] Generate charts...")
    plot_prediction_charts(predictions, dates_test, close_test)
    plot_confusion_matrices(predictions, y_test)
    plot_accuracy_comparison(results_df)
    plot_backtest_equity(backtest_results, rl_equity=rl_equity, dates_test=dates_test)

    # save risk summary
    plot_risk_summary(backtest_results, rl_equity=rl_equity)

    # save ke csv
    print("\n[SAVE] Menyimpan hasil...")
    save_comparison_csv(results_df, backtest_results)

    print("\n" + "=" * 60)
    print("  SELESAI! Check file .png dan .csv buat hasilnya.")
    print("=" * 60)
