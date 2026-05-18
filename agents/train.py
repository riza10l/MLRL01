
import os
import sys
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import build_all_features, get_feature_columns
from features.regime_detection import add_regime_features
from env.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from risk.risk_manager import RiskManager

SCALED_MODELS = {"Logistic Regression", "SVM"}

# Embargo size: max rolling window used in features
EMBARGO_BARS = 60

def load_latest_data(data_dir="../jupiter"):
    
    pattern = os.path.join(data_dir, "data*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    # Filter to only dataNN.csv (numeric suffix), exclude data_saham etc
    numbered = [f for f in files if re.search(r'data\d+\.csv$', os.path.basename(f))]
    if numbered:
        # Pick highest numbered file
        latest = max(numbered, key=lambda f: int(re.search(r'data(\d+)\.csv$', os.path.basename(f)).group(1)))
    else:
        latest = max(files, key=os.path.getctime)
    print(f"[DATA] Using file: {latest}")
    df = pd.read_csv(latest)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
    print(f"[DATA] Rows: {len(df):,} | {df['date'].min().date()} -> {df['date'].max().date()}")
    return df

def split_data(df, feature_cols, train_ratio=0.80, embargo=EMBARGO_BARS):
    
    split_idx = int(len(df) * train_ratio)

    # Apply embargo: skip `embargo` bars after train end
    test_start = split_idx + embargo

    if test_start >= len(df):
        print(f"[WARN] Embargo ({embargo}) exceeds remaining data. Reducing.")
        test_start = split_idx + 10

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test = X.iloc[:split_idx], X.iloc[test_start:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[test_start:]
    dates_test = df["date"].iloc[test_start:].reset_index(drop=True)
    close_test = df["close"].iloc[test_start:].reset_index(drop=True)

    # Scaler fitted on TRAIN only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"[SPLIT] Train: {len(X_train):,} | Embargo: {embargo} bars | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, dates_test, close_test, scaler

def get_ml_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

def train_ml_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    models = get_ml_models()
    predictions = {}
    results = []
    trained = {}
    print("\n[ML] Training semua model...")
    for name, model in models.items():
        Xtr = X_train_sc if name in SCALED_MODELS else X_train
        Xte = X_test_sc if name in SCALED_MODELS else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        predictions[name] = y_pred
        trained[name] = model
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec})
        print(f"  {name:<22} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")

    # Sanity check: warn if any model > 60% accuracy
    for r in results:
        if r["Accuracy"] > 0.60:
            print(f"  [WARN] {r['Model']} accuracy={r['Accuracy']:.1%} — "
                  f"suspiciously high for daily prediction. Check for leakage!")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    results_df.index += 1
    return predictions, results_df, trained

def train_rl_agent(df_train, feature_cols, timesteps=100_000, use_lstm=False, save_path=None):
    
    env = TradingEnv(df_train, feature_columns=feature_cols)
    agent = PPOAgent(env, use_lstm=use_lstm)
    agent.train(total_timesteps=timesteps)
    if save_path:
        agent.save(save_path)
    return agent

def walk_forward_validation(df, feature_cols, train_years=3, test_years=1,
                           step_years=1, embargo=EMBARGO_BARS):
    
    print("\n[WALK-FORWARD] Starting purged walk-forward validation...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    min_year = df["date"].dt.year.min()
    max_year = df["date"].dt.year.max()

    fold_results = []
    fold = 0

    for start_year in range(min_year, max_year - train_years - test_years + 2, step_years):
        train_end_year = start_year + train_years
        test_end_year = train_end_year + test_years

        train_mask = (df["date"].dt.year >= start_year) & (df["date"].dt.year < train_end_year)
        test_mask = (df["date"].dt.year >= train_end_year) & (df["date"].dt.year < test_end_year)

        df_train_raw = df[train_mask].reset_index(drop=True)
        df_test_raw = df[test_mask].reset_index(drop=True)

        if len(df_train_raw) < 200 or len(df_test_raw) < 50:
            continue

        # Apply embargo: remove first `embargo` bars from test
        df_test = df_test_raw.iloc[embargo:].reset_index(drop=True)
        if len(df_test) < 30:
            continue

        fold += 1
        print(f"\n  Fold {fold}: Train {start_year}-{train_end_year-1} | "
              f"Test {train_end_year}-{test_end_year-1} (embargo={embargo})")
        print(f"    Train rows: {len(df_train_raw):,} | Test rows: {len(df_test):,}")

        # Train RL
        env_train = TradingEnv(df_train_raw, feature_columns=feature_cols)
        agent = PPOAgent(env_train, verbose=0)
        agent.train(total_timesteps=50_000)

        # Evaluate on test
        env_test = TradingEnv(df_test, feature_columns=feature_cols)
        equity, stats, _ = agent.evaluate(env_test)

        metrics = RiskManager.compute_all_metrics(equity)
        metrics["fold"] = fold
        metrics["train_period"] = f"{start_year}-{train_end_year-1}"
        metrics["test_period"] = f"{train_end_year}-{test_end_year-1}"
        metrics["embargo"] = embargo
        fold_results.append(metrics)

        print(f"    Return: {metrics['total_return']:+.2%} | "
              f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"MaxDD: {metrics['max_drawdown']:.2%}")

    if fold_results:
        summary = pd.DataFrame(fold_results)
        avg_sharpe = summary["sharpe_ratio"].mean()
        std_sharpe = summary["sharpe_ratio"].std()
        avg_return = summary["total_return"].mean()
        print(f"\n[WALK-FORWARD] Summary: {len(fold_results)} folds")
        print(f"  Avg Return: {avg_return:+.2%}")
        print(f"  Avg Sharpe: {avg_sharpe:.3f} +/- {std_sharpe:.3f}")
        print(f"  Sharpe stable? {'YES' if std_sharpe < 0.5 else 'NEEDS REVIEW'}")
        return summary
    else:
        print("[WALK-FORWARD] Not enough data for walk-forward validation")
        return pd.DataFrame()