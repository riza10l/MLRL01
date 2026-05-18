#backtest mesin mk4

import numpy as np
import pandas as pd
from risk.risk_manager import RiskManager
from backtest.metrics import BacktestMetrics

class BacktestEngine:
    def __init__(self, initial_capital=100_000, stop_loss=0.01,
                 take_profit=0.015, fee=0.001, spread=0.0003,
                 slippage=0.0002):
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.fee = fee
        self.spread = spread
        self.slippage = slippage
        self.total_cost_rate = fee + spread + slippage

    def run_ml_backtest(self, predictions, close_test, dates_test=None):
        
        results = {}
        print("\n[BACKTEST] Running realistic ML backtests...")
        for name, preds in predictions.items():
            equity, metrics = self._backtest_single(preds, close_test.values)
            metrics["equity"] = equity
            results[name] = metrics
            print(f"  {name:<22} ret={metrics['total_return']:+.2%} "
                  f"sharpe={metrics['sharpe_ratio']:.3f} "
                  f"trades={metrics['n_trades']} "
                  f"costs=${metrics['total_costs']:,.0f}")
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
            price_next = closes[min(i + 1, len(closes) - 1)]

            if pos == 1:
                pnl_pct = (price - entry) / (entry + 1e-10)

                # Stop-loss check
                if pnl_pct <= -self.stop_loss:
                    c = cap * self.total_cost_rate
                    realized_pnl = -self.stop_loss
                    cap *= (1 + realized_pnl)
                    cap -= c
                    costs += c
                    trades.append(realized_pnl)
                    pos = 0

                # Take-profit check
                elif pnl_pct >= self.take_profit:
                    c = cap * self.total_cost_rate
                    realized_pnl = self.take_profit
                    cap *= (1 + realized_pnl)
                    cap -= c
                    costs += c
                    trades.append(realized_pnl)
                    pos = 0

                # Model signals exit
                elif preds[i] == 0:
                    c = cap * self.total_cost_rate
                    cap *= (1 + pnl_pct)
                    cap -= c
                    costs += c
                    trades.append(pnl_pct)
                    pos = 0

            else:
                if preds[i] == 1:
                    # Entry cost
                    c = cap * self.total_cost_rate
                    cap -= c
                    costs += c
                    pos = 1
                    # Enter at next bar (execution delay)
                    entry = price_next

            # Update equity
            if pos == 1 and entry > 0:
                current_price = closes[min(i + 1, len(closes) - 1)]
                unrealized = (current_price - entry) / (entry + 1e-10)
                equity.append(cap * (1 + unrealized))
            else:
                equity.append(cap)

        equity = np.array(equity)
        metrics = BacktestMetrics.compute(equity, trades, self.initial_capital, costs, len(trades))
        return equity, metrics

    @staticmethod
    def detect_anomalous_candles(df, zscore_threshold=5.0):
        
        anomalies = []

        # Check for zero/negative prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                bad = df[col] <= 0
                if bad.any():
                    anomalies.append(bad)
                    print(f"  [ANOMALY] {bad.sum()} rows with {col} <= 0")

        # Check for extreme returns
        if 'close' in df.columns:
            ret = df['close'].pct_change().abs()
            extreme = ret > 0.15  # >15% daily move
            if extreme.any():
                anomalies.append(extreme)
                print(f"  [ANOMALY] {extreme.sum()} rows with >15% daily move")

        # Check for volume spikes (z-score)
        if 'volume' in df.columns:
            vol_z = (df['volume'] - df['volume'].rolling(50).mean()) / (df['volume'].rolling(50).std() + 1e-10)
            extreme_vol = vol_z.abs() > zscore_threshold
            if extreme_vol.any():
                print(f"  [ANOMALY] {extreme_vol.sum()} rows with extreme volume (z>{zscore_threshold})")

        # Check for high==low (no trading)
        if 'high' in df.columns and 'low' in df.columns:
            no_range = df['high'] == df['low']
            if no_range.any():
                print(f"  [ANOMALY] {no_range.sum()} rows with high==low (no trading)")

        if anomalies:
            combined = pd.concat(anomalies, axis=1).any(axis=1)
            clean_mask = ~combined
            print(f"  [ANOMALY] Total anomalous rows: {combined.sum()}")
            return clean_mask
        else:
            print("  [ANOMALY] No anomalies detected")
            return pd.Series(True, index=df.index)