#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariate Time Series Forecasting with Exogenous Variables
Architecture: TCN + Attention + GRU (Probabilistic)
Includes: Synthetic data, preprocessing, custom loss, walk-forward CV, 
          baselines, Diebold-Mariano test, and diagnostics.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
import statsmodels.graphics.tsaplots
# from arch.unitroot import tsadf  # Optional, used for sanity checks
from scipy.stats import norm, probplot
from scipy.signal import correlate
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# =============================================================================
# 1. DATA GENERATION
# =============================================================================
class SyntheticDataGenerator:
    """Generates multivariate time series with trends, seasonality, missing values, and outliers."""
    
    def __init__(self, n_samples: int = 3650, freq: str = "D"):
        self.n_samples = n_samples
        self.freq = freq
        
    def generate(self) -> pd.DataFrame:
        dates = pd.date_range(start="2020-01-01", periods=self.n_samples, freq=self.freq)
        t = np.arange(self.n_samples)
        
        # Trend + Multiple Seasonality
        trend = 0.05 * t + 100
        seasonal_daily = 10 * np.sin(2 * np.pi * t / 7)
        seasonal_yearly = 20 * np.sin(2 * np.pi * t / 365)
        noise = np.random.normal(0, 2, self.n_samples)
        
        # Endogenous (coupled)
        y1 = trend + seasonal_daily + seasonal_yearly + noise
        y2 = 0.7 * y1 + 0.3 * np.roll(y1, shift=3) + noise * 1.5
        y3 = 0.5 * y2 - 0.2 * trend + noise * 0.8
        
        # Exogenous (random + seasonal)
        x1 = np.random.uniform(-1, 1, self.n_samples) + 5 * np.cos(2 * np.pi * t / 30)
        x2 = np.random.normal(0, 1, self.n_samples) + 3 * np.sin(2 * np.pi * t / 14)
        
        df = pd.DataFrame({
            "y1": y1, "y2": y2, "y3": y3,
            "x1": x1, "x2": x2
        }, index=dates)
        
        # Missing Values (~5%)
        mask_missing = np.random.rand(*df.shape) < 0.05
        df[mask_missing] = np.nan
        
        # Outliers (~2% spikes)
        n_outliers = int(0.02 * self.n_samples)
        outlier_idx = np.random.choice(self.n_samples, n_outliers, replace=False)
        cols_endog = ["y1", "y2", "y3"]
        for i in outlier_idx:
            col = np.random.choice(cols_endog)
            df.loc[df.index[i], col] *= np.random.uniform(2.5, 4.0)
            
        return df


# =============================================================================
# 2. PREPROCESSING PIPELINE
# =============================================================================
class TimeSeriesPreprocessor:
    """Handles imputation, outlier correction, differencing, STL, and feature engineering."""
    
    def __init__(self, target_cols: list[str], exog_cols: list[str]):
        self.target_cols = target_cols
        self.exog_cols = exog_cols
        self.scaler_target = StandardScaler()
        self.scaler_exog = StandardScaler()
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        self.iqr_factors = {}
        
    def fit_transform(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        print("  - Imputing missing values...")
        df = self._impute_missing(df)
        print("  - Handling outliers...")
        df = self._handle_outliers(df)
        print("  - Applying differencing...")
        df = self._apply_differencing(df)
        print("  - Extracting STL features...")
        df = self._extract_stl_features(df)
        print("  - Engineering features...")
        df = self._engineer_features(df)
        print("  - Scaling features...")
        df = self._scale_features(df)
        return df
    
    def inverse_transform_targets(self, df_scaled: pd.DataFrame, diff_orders: dict) -> pd.DataFrame:
        # Reverse scaling & differencing for targets
        df_inv = df_scaled.copy()
        for col in self.target_cols:
            vals = df_inv[f"{col}_scaled"]
            vals = vals * self.scaler_target.scale_[self.target_cols.index(col)] + self.scaler_target.mean_[self.target_cols.index(col)]
            # Integrate differences
            diffs = self.iqr_factors[col]["diff_order"]
            for _ in range(diffs):
                vals = np.cumsum(vals)
            df_inv[f"{col}_pred"] = vals
        return df_inv

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include="number")
        num_imputed = self.imputer.fit_transform(numeric)
        for col, vals in zip(numeric.columns, num_imputed.T):
            df[col] = vals
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)
            self.iqr_factors.setdefault(col, {})["diff_order"] = 0  # Initializing tracking
        return df

    def _apply_differencing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_cols:
            df[f"{col}_diff"] = df[col].diff().dropna()
            self.iqr_factors.setdefault(col, {})["diff_order"] = 1
        return df.dropna()

    def _extract_stl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.target_cols:
            try:
                stl = STL(df[f"{col}_diff"], period=365, robust=True)
                res = stl.fit()
                df_out[f"{col}_seasonal"] = res.seasonal
                df_out[f"{col}_resid"] = res.resid
                df_out[f"{col}_trend"] = res.trend
            except Exception:
                df_out[f"{col}_seasonal"] = 0
                df_out[f"{col}_resid"] = 0
                df_out[f"{col}_trend"] = 0
        return df_out.drop(columns=[c.replace("_diff","") for c in self.target_cols])

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_vals = [1, 2, 3, 4, 5, 6, 7]
        for col in self.target_cols:
            base = f"{col}_diff"
            for k in lag_vals:
                df[f"{col}_lag{k}"] = df[base].shift(k)
            for w in [3, 7]:
                df[f"{col}_roll_mean_{w}"] = df[base].rolling(w).mean()
                df[f"{col}_roll_std_{w}"] = df[base].rolling(w).std()
                
        # Time features
        idx = df.index
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["day_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        df["quarter"] = idx.quarter
        
        return df.dropna()

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_cols = [c for c in df.columns if c not in self.target_cols + self.exog_cols + ["hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos","quarter"]]
        df[feat_cols] = self.scaler_target.fit_transform(df[feat_cols])
        # Scale exog separately
        df[self.exog_cols] = self.scaler_exog.fit_transform(df[self.exog_cols])
        return df


# =============================================================================
# 3. PYTORCH DATASET & DATA LOADER
# =============================================================================
class TSForecastDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for sequential sampling."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.n_samples = len(X) - lookback - horizon + 1
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        return self.X[idx : idx + self.lookback], self.y[idx + self.lookback : idx + self.lookback + self.horizon]


# =============================================================================
# 4. MODEL ARCHITECTURE: TCN + ATTENTION + GRU
# =============================================================================
class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        out = self.conv(x.transpose(1, 2))
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.norm(out) + x  # Residual
        return self.relu(self.dropout(out))

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.norm2(x + self.ff(x))

class HybridTCNAttGRU(nn.Module):
    def __init__(self, input_dim: int, horizon: int, n_targets: int, hidden_dims: list[int] = [64, 32], n_tcn_layers: int = 2, d_model: int = 32):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.input_proj = nn.Linear(input_dim, d_model)
        dilations = [2**i for i in range(n_tcn_layers)]
        self.tcn = nn.ModuleList([
            TCNBlock(d_model if i==0 else d_model, d_model, dilation=d) 
            for i, d in enumerate(dilations)
        ])
        self.attention = SelfAttention(d_model)
        self.gru = nn.GRU(d_model, hidden_dims[-1], batch_first=True)
        self.head = nn.Linear(hidden_dims[-1], horizon * n_targets * 2)  # mu, sigma for each step/target
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.tcn:
            x = block(x)
        x = self.attention(x)
        _, h_n = self.gru(x)
        h = h_n[-1]  # Last layer hidden state
        out = self.head(h)
        out = out.view(-1, self.horizon, self.n_targets, 2)
        mu, raw_sigma = out[..., 0], out[..., 1]
        sigma = torch.nn.functional.softplus(raw_sigma) + 1e-4  # Softplus is better for variance
        return mu, sigma


# =============================================================================
# 5. LOSS FUNCTION
# =============================================================================
class ProbabilisticHybridLoss(nn.Module):
    """Combined Negative Log-Likelihood + Pinball Regularization"""
    def __init__(self, alphas: list[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.alphas = alphas
        
    def forward(self, mu, sigma, y_true):
        # NLL for Gaussian
        dist = torch.distributions.Normal(mu, sigma)
        nll = -dist.log_prob(y_true).mean()
        
        # Pinball loss on quantiles derived from Gaussian parameters
        pinball_loss = torch.tensor(0.0, device=mu.device)
        z_scores = torch.tensor([norm.ppf(q) for q in self.alphas], device=mu.device)
        for i, q in enumerate(self.alphas):
            q_pred = mu + sigma * z_scores[i]
            error = y_true - q_pred
            pinball_loss += torch.mean((error) * (q - (error < 0).float()))
            
        lam = 0.2  # Regularization weight
        return nll + lam * pinball_loss


# =============================================================================
# 6. TRAINER
# =============================================================================
class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, lr: float = 1e-3, grad_clip: float = 5.0):
        self.model = model
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.device = next(model.parameters()).device
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            mu, sigma = self.model(X_batch)
            loss = self.criterion(mu, sigma, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        mses, crps_vals = [], []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                mu, sigma = self.model(X)
                
                # MSE: Reshape to 2D (N*T, F) for sklearn
                y_np = y.cpu().numpy()
                mu_np = mu.cpu().numpy()
                y_2d = y_np.reshape(-1, y_np.shape[-1])
                mu_2d = mu_np.reshape(-1, mu_np.shape[-1])
                mses.append(mean_squared_error(y_2d, mu_2d))
                
                # CRPS (Gaussian closed-form)
                z = (y - mu) / sigma
                z_cpu = z.cpu().numpy()
                sigma_cpu = sigma.cpu().numpy()
                # Use scipy functions on numpy arrays
                crps = sigma_cpu * (z_cpu * (2 * norm.cdf(z_cpu) - 1) + 2 * norm.pdf(z_cpu) - 1/np.sqrt(np.pi))
                # Note: the formula for CRPS of a Gaussian is σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
                crps_vals.append(np.mean(crps))
        return np.mean(mses), np.mean(crps_vals)


# =============================================================================
# 7. EVALUATOR & BASELINES
# =============================================================================
class Evaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray) -> dict:
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MASE
        naive_errors = np.abs(np.diff(train_series[:len(y_true)+1]))
        mase = mae / (np.mean(naive_errors) + 1e-6)
        
        # sMAPE
        denom = np.abs(y_true) + np.abs(y_pred)
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (denom + 1e-6))
        
        # CRPS
        z = (y_true - y_pred) / (np.std(y_true) + 1e-4) # Approx sigma for eval
        crps_val = np.mean((y_true - y_pred) * (2 * norm.cdf(z) - 1) - 2 * norm.pdf(z) + 2 / np.sqrt(np.pi))
        
        return {"MAE": mae, "RMSE": rmse, "MASE": mase, "sMAPE": smape, "CRPS": crps_val}
    
    @staticmethod
    def diebold_mariano(e1: np.ndarray, e2: np.ndarray, nobs_adj: bool = True) -> float:
        """Standard DM test statistic comparing squared errors of two models."""
        d = e1**2 - e2**2
        n = len(d)
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1) if nobs_adj else np.var(d)
        dm_stat = np.sqrt(n) * mean_d / np.sqrt(var_d)
        return abs(dm_stat) > 1.96  # p < 0.05 rejection threshold

    @classmethod
    def run_walk_forward(cls, df_train: np.ndarray, y_train: np.ndarray, 
                         df_test: np.ndarray, y_test: np.ndarray, 
                         lookback: int, horizons: list[int]):
        results = {}
        train_series = df_train[:, 0] # Use first target as reference for MASE
        
        for h in horizons:
            print(f"\n--- Horizon {h} ---")
            # 1. Naive: predicting the value at the start of the horizon
            naive_pred = y_test[:, 0, 0] # Target 0
            metrics_naive = cls.calculate_metrics(y_test[:, h-1, 0], naive_pred, train_series)
            results[f"Naive_H{h}"] = metrics_naive
            
            # 2. ARIMA (simplified per horizon, target 0)
            arima_preds = np.zeros(len(y_test))
            for i in range(max(0, len(y_test)-50), len(y_test)):
                hist = np.concatenate([df_train[:, 0][-lookback:], y_test[:i, 0, 0]])
                try:
                    model_ar = ARIMA(hist[-365:], order=(1,1,1)).fit()
                    arima_preds[i] = model_ar.forecast(steps=1)[0]
                except:
                    arima_preds[i] = np.mean(hist[-10:])
            metrics_arima = cls.calculate_metrics(y_test[:, h-1, 0], arima_preds, train_series)
            results[f"ARIMA_H{h}"] = metrics_arima
            
            # 3. XGBoost (target 0)
            xgb_dataset = xgb.DMatrix(df_test, label=y_test[:, h-1, 0])
            params = {"max_depth": 6, "eta": 0.01, "objective": "reg:squarederror", "eval_metric": "rmse"}
            xgb_model = xgb.train(params, xgb_dataset, num_boost_round=100) # Faster
            xgb_pred = xgb_model.predict(xgb_dataset)
            metrics_xgb = cls.calculate_metrics(y_test[:, h-1, 0], xgb_pred, train_series)
            results[f"XGBoost_H{h}"] = metrics_xgb
            
        return results


# =============================================================================
# 8. VISUALIZER
# =============================================================================
class Visualizer:
    @staticmethod
    def plot_actual_vs_predicted(actual: np.ndarray, pred_mu: np.ndarray, pred_sigma: np.ndarray, horizon: int):
        plt.figure(figsize=(12, 5))
        plt.plot(actual, label="Actual", alpha=0.7)
        plt.plot(pred_mu, label="Predicted Mean", color="red")
        plt.fill_between(range(len(actual)), pred_mu - 1.64*pred_sigma, pred_mu + 1.64*pred_sigma, 
                         color="red", alpha=0.2, label="90% Confidence Interval")
        plt.title(f"Actual vs Predicted (Horizon {horizon})")
        plt.legend(); plt.grid(True); plt.show()
        
    @staticmethod
    def feature_importance(X_val: np.ndarray, y_val: np.ndarray, feature_names: list[str]):
        model_xgb = xgb.XGBRegressor(n_estimators=50)
        model_xgb.fit(X_val, y_val)
        perm_imp = permutation_importance(
            model_xgb, 
            X_val, y_val, n_repeats=5, random_state=42
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=perm_imp.importances_mean, y=feature_names)
        plt.title("Permutation Feature Importance"); plt.xlabel("Mean Decrease in Accuracy"); plt.tight_layout(); plt.show()
        
    @staticmethod
    def residual_diagnostics(residuals: np.ndarray):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(residuals, bins=30, density=True); axes[0].set_title("Residual Histogram")
        statsmodels.graphics.tsaplots.plot_acf(residuals, ax=axes[1]); axes[1].set_title("ACF Plot")
        probplot(residuals, dist="norm", plot=axes[2]); axes[2].set_title("Q-Q Plot")
        plt.tight_layout(); plt.show()


# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multivariate Probabilistic TS Forecasting")
    parser.add_argument("--n_samples", type=int, default=3650, help="Dataset length")
    parser.add_argument("--lookback", type=int, default=30, help="Sequence length")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--output_horizons", nargs="+", type=int, default=[1, 3, 6, 12], help="Evaluation horizons")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32], help="Hidden dimensions")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running on {device}")
    
    # 1. Generate & Preprocess
    gen = SyntheticDataGenerator(n_samples=args.n_samples)
    df_full = gen.generate()
    preprocessor = TimeSeriesPreprocessor(target_cols=["y1", "y2", "y3"], exog_cols=["x1", "x2"])
    df_proc = preprocessor.fit_transform(df_full)
    
    # Extract tensors
    target_cols = [f"{c}_diff" for c in ["y1", "y2", "y3"]]
    exog_cols = ["x1", "x2"]
    feat_cols = [c for c in df_proc.columns if c not in target_cols + exog_cols + ["hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos","quarter"]]
    
    X_all = df_proc[feat_cols + exog_cols].values
    y_all = df_proc[target_cols].values
    X_all = np.nan_to_num(X_all, nan=0.0)
    
    # Train/Val/Test split (Time-based)
    split_tr, split_val = int(0.7 * len(X_all)), int(0.85 * len(X_all))
    X_tr, X_val, X_te = X_all[:split_tr], X_all[split_tr:split_val], X_all[split_val:]
    y_tr, y_val, y_te = y_all[:split_tr], y_all[split_tr:split_val], y_all[split_val:]
    
    # Build Datasets
    ds_tr = TSForecastDataset(X_tr, y_tr, args.lookback, args.horizon)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 2. Initialize Model & Training
    input_dim = X_tr.shape[1]
    n_targets = y_tr.shape[1] if len(y_tr.shape) > 1 else 1 # Fallback for single target
    model = HybridTCNAttGRU(input_dim=input_dim, horizon=args.horizon, n_targets=len(target_cols)).to(device)
    criterion = ProbabilisticHybridLoss()
    trainer = Trainer(model, criterion, lr=args.lr, grad_clip=5.0)
    
    # Walk-forward simulation wrapper for training
    best_crps = np.inf
    patience_counter = 0
    history = []
    
    print("\nTraining...")
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(dl_tr)
        val_mse, val_crps = trainer.evaluate(DataLoader(TSForecastDataset(X_val, y_val, args.lookback, args.horizon), batch_size=args.batch_size))
        trainer.scheduler.step(val_crps)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse, "val_crps": val_crps})
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val CRPS: {val_crps:.4f}")
        
        if val_crps < best_crps:
            best_crps = val_crps
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
                
    # Load best
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    
    # 3. Evaluate Multi-Horizon
    print("\nEvaluating baselines & hybrid model...")
    test_ds = TSForecastDataset(X_te, y_te, args.lookback, args.horizon)
    dl_te = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    y_true_te = []
    y_pred_te_mu = []
    y_pred_te_sigma = []
    with torch.no_grad():
        for X_b, y_b in dl_te:
            X_b, y_b = X_b.to(device), y_b.to(device)
            mu, sigma = model(X_b)
            y_true_te.append(y_b.cpu().numpy())
            y_pred_te_mu.append(mu.cpu().numpy())
            y_pred_te_sigma.append(sigma.cpu().numpy())
            
    y_true_te = np.vstack(y_true_te)
    y_pred_te_mu = np.vstack(y_pred_te_mu)
    y_pred_te_sigma = np.vstack(y_pred_te_sigma)
    
    # Run baseline comparison
    # Align features with the number of samples in y_true_te
    X_te_aligned = X_te[:len(y_true_te)]
    baseline_results = Evaluator.run_walk_forward(X_tr, y_tr, X_te_aligned, y_true_te, args.lookback, args.output_horizons)
    
    # 4. Print & DM Test
    print("\nPERFORMANCE SUMMARY:")
    df_res = pd.DataFrame(baseline_results).T
    print(df_res.round(3))
    
    for h in args.output_horizons:
        # Compare Hybrid vs Naive for target 0
        h_idx = h - 1
        e_hybrid = np.abs(y_true_te[:, h_idx, 0] - y_pred_te_mu[:, h_idx, 0])
        # Naive: last known value of target 0 (at t=0 in the horizon window)
        e_naive = np.abs(y_true_te[:, h_idx, 0] - y_true_te[:, 0, 0])
        sig_diff = Evaluator.diebold_mariano(e_hybrid, e_naive)
        print(f"Horizon {h}: DM Test (Hybrid vs Naive) Significant? {sig_diff}")
        
    # 5. Visualizations
    print("\nGenerating visualizations...")
    V = Visualizer()
    V.plot_actual_vs_predicted(y_true_te[:, 0, 0], y_pred_te_mu[:, 0, 0], y_pred_te_sigma[:, 0, 0], 1)
    
    residuals = y_true_te[:, 0, 0] - y_pred_te_mu[:, 0, 0]
    V.residual_diagnostics(residuals)
    
    # Feature importance (using last slice as example)
    feat_names = feat_cols + exog_cols
    V.feature_importance(X_val[:1000], y_val[:1000, 0], feat_names)
    
    print("\nPipeline complete. Artifacts saved: best_model.pt, metrics table.")

if __name__ == "__main__":
    main()