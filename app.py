"""
MLRL01 — Streamlit Dashboard
Gold Futures Quantitative Trading Engine
Converted from CLI main.py → Interactive Web Dashboard
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Path setup (same as main.py) ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import glob

# ── Internal module imports ───────────────────────────────────────────────────
from agents.train import (
    load_latest_data, split_data, train_ml_models,
    train_rl_agent, walk_forward_validation, EMBARGO_BARS,
)
from agents.evaluate import (
    plot_equity_curves, plot_prediction_charts,
    plot_confusion_matrices, plot_accuracy_comparison,
    plot_walk_forward, save_risk_summary, save_comparison_csv,
)
from features.features import build_production_features, get_production_feature_columns
from env.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from backtest.engine import BacktestEngine
from backtest.benchmarks import BenchmarkRunner
from backtest.metrics import BacktestMetrics
from risk.risk_manager import RiskManager
from monte_carlo.simulator import MonteCarloSimulator

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR_PLOTS   = os.path.join(BASE_DIR, "results", "plots")
SAVE_DIR_REPORTS = os.path.join(BASE_DIR, "results", "reports")
SAVE_DIR_MC      = os.path.join(BASE_DIR, "results", "monte_carlo")
MODEL_SAVE_DIR   = os.path.join(BASE_DIR, "models", "saved_models")
DATA_DIR         = "jupiter" if os.path.exists("jupiter") else os.path.join(BASE_DIR, "jupiter")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MLRL01 — Gold Quant Engine",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — Dark-Gold premium theme
st.markdown("""
<style>
    /* ── Import premium font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root variables ── */
    :root {
        --gold-primary: #D4A843;
        --gold-light: #F0D78C;
        --gold-dark: #A67C2E;
        --gold-glow: rgba(212, 168, 67, 0.25);
        --bg-dark: #0E1117;
        --bg-card: #1A1D26;
        --bg-card-hover: #22252F;
        --text-primary: #E8E6E3;
        --text-secondary: #9CA3AF;
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
        --info: #3B82F6;
    }

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1A1D26 0%, #2A2520 50%, #1A1D26 100%);
        border: 1px solid rgba(212, 168, 67, 0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #D4A843, #F0D78C, #D4A843, transparent);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F0D78C 0%, #D4A843 50%, #A67C2E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1A1D26, #22252F);
        border: 1px solid rgba(212, 168, 67, 0.15);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(212, 168, 67, 0.4);
        box-shadow: 0 4px 20px rgba(212, 168, 67, 0.1);
        transform: translateY(-2px);
    }
    .metric-label {
        color: #9CA3AF;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-positive { color: #10B981; }
    .metric-negative { color: #EF4444; }
    .metric-neutral  { color: #D4A843; }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-clean {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ── Section header ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #D4A843;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(212, 168, 67, 0.2);
    }

    /* ── Image container ── */
    .plot-container {
        background: #1A1D26;
        border: 1px solid rgba(212, 168, 67, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13161D 0%, #0E1117 100%);
        border-right: 1px solid rgba(212, 168, 67, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #D4A843 !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
    }

    /* ── Button ── */
    .stButton > button[kind="primary"],
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #D4A843 0%, #A67C2E 100%) !important;
        color: #0E1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 4px 25px rgba(212, 168, 67, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Divider ── */
    .gold-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212,168,67,0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_metric_card(label, value, fmt="str", prefix="", suffix=""):
    """Render a styled metric card."""
    if fmt == "pct":
        display = f"{prefix}{value:+.2%}{suffix}"
        css_class = "metric-positive" if value > 0 else "metric-negative"
    elif fmt == "float":
        display = f"{prefix}{value:.3f}{suffix}"
        css_class = "metric-positive" if value > 0 else ("metric-negative" if value < 0 else "metric-neutral")
    elif fmt == "int":
        display = f"{prefix}{int(value):,}{suffix}"
        css_class = "metric-neutral"
    else:
        display = f"{prefix}{value}{suffix}"
        css_class = "metric-neutral"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{display}</div>
    </div>
    """, unsafe_allow_html=True)


def show_image_safe(path, caption=None):
    """Display an image if file exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"📁 File not found: `{os.path.basename(path)}`")


def format_mc_report(report):
    """Format Monte Carlo report as a readable dict for display."""
    return {
        "Simulations": f"{report.get('n_simulations', 0):,}",
        "Mean Return": f"{report.get('mean_return', 0):+.2%}",
        "Median Return": f"{report.get('median_return', 0):+.2%}",
        "P(Positive)": f"{report.get('prob_positive', 0):.1%}",
        "P(Ruin >10%)": f"{report.get('prob_ruin_10pct', 0):.1%}",
        "P(Ruin >20%)": f"{report.get('prob_ruin_20pct', 0):.1%}",
        "Mean Max DD": f"{report.get('mean_max_dd', 0):.2%}",
        "Worst Max DD": f"{report.get('worst_max_dd', 0):.2%}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🥇 MLRL01")
    st.markdown('<p style="color:#9CA3AF; font-size:0.85rem;">Gold Futures Quant Engine</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Training Config")

    TRAIN_RATIO = st.slider(
        "Train Ratio", min_value=0.50, max_value=0.95,
        value=0.80, step=0.05, help="Fraction of data used for training"
    )
    RL_TIMESTEPS = st.number_input(
        "RL Timesteps", min_value=10_000, max_value=500_000,
        value=50_000, step=10_000, help="PPO training timesteps"
    )
    USE_LSTM = st.checkbox("Use LSTM Policy", value=False,
                           help="Use RecurrentPPO (requires sb3-contrib)")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Target Engineering")

    TARGET_HORIZON = st.number_input(
        "Target Horizon (days)", min_value=1, max_value=30,
        value=5, step=1
    )
    TARGET_THRESHOLD = st.number_input(
        "Target Threshold", min_value=0.001, max_value=0.050,
        value=0.005, step=0.001, format="%.3f"
    )
    TARGET_METHOD = "triple_barrier"

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔬 Validation")

    RUN_WALK_FWD = st.checkbox("Walk-Forward Validation", value=True)
    RUN_MONTE_CARLO = st.checkbox("Monte Carlo Simulation", value=True)
    MC_SIMULATIONS = st.number_input(
        "MC Simulations", min_value=100, max_value=10_000,
        value=1_000, step=100,
        disabled=not RUN_MONTE_CARLO
    )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#6B7280; font-size:0.7rem; text-align:center;">'
        'MLRL01 V4 — Leakage-Free Engine<br>Gold Futures (GC=F)</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🥇 MLRL01 — Gold Quant Engine</h1>
    <p class="hero-subtitle">
        V4 Leakage-Free Quantitative Trading Engine · Gold Futures (GC=F) · Professional Audit Applied
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

start_clicked = st.button("🚀  Start Engine", use_container_width=True)

if start_clicked:
    # Create output dirs
    for d in [SAVE_DIR_PLOTS, SAVE_DIR_REPORTS, SAVE_DIR_MC, MODEL_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    with st.status("⏳ Running MLRL01 Engine...", expanded=True) as status:

        # ── Stage 1: Load Data ────────────────────────────────────────────
        st.write("📥 **Stage 1/7** — Loading data...")
        try:
            df_raw = load_latest_data(DATA_DIR)
            st.write(f"✅ Loaded {len(df_raw):,} rows "
                     f"({df_raw['date'].min().date()} → {df_raw['date'].max().date()})")
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            st.stop()

        # ── Stage 2: Anomaly Scan & Feature Engineering ───────────────────
        st.write("🔧 **Stage 2/7** — Feature engineering...")
        try:
            clean_mask = BacktestEngine.detect_anomalous_candles(df_raw)
            n_removed = (~clean_mask).sum()
            if n_removed > 0:
                df_raw = df_raw[clean_mask].reset_index(drop=True)
                st.write(f"⚠️ Removed {n_removed} anomalous candles")

            df = build_production_features(
                df_raw,
                target_horizon=TARGET_HORIZON,
                target_threshold=TARGET_THRESHOLD,
                target_method=TARGET_METHOD,
            )
            feature_cols = get_production_feature_columns(df)
            st.write(f"✅ Built {len(feature_cols)} features (leakage-free)")
        except Exception as e:
            st.error(f"❌ Feature engineering failed: {e}")
            st.stop()

        # ── Stage 3: Split & Train ML ─────────────────────────────────────
        st.write("🤖 **Stage 3/7** — Training ML models...")
        try:
            (X_train, X_test, y_train, y_test,
             X_train_sc, X_test_sc,
             dates_test, close_test, scaler) = split_data(df, feature_cols, TRAIN_RATIO)

            predictions, results_df, trained_models = train_ml_models(
                X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
            )
            st.write(f"✅ Trained {len(trained_models)} ML models")
        except Exception as e:
            st.error(f"❌ ML training failed: {e}")
            st.stop()

        # ── Stage 4: Train RL Agent ───────────────────────────────────────
        st.write("🧠 **Stage 4/7** — Training RL agent (PPO)...")
        try:
            split_idx = int(len(df) * TRAIN_RATIO)
            df_train_slice = df.iloc[:split_idx].reset_index(drop=True)
            test_start = split_idx + EMBARGO_BARS
            df_test_slice = df.iloc[test_start:].reset_index(drop=True)

            save_path = os.path.join(MODEL_SAVE_DIR, "ppo_trading_gold_v4")
            agent = train_rl_agent(
                df_train_slice, feature_cols,
                timesteps=RL_TIMESTEPS,
                use_lstm=USE_LSTM,
                save_path=save_path,
            )

            env_test = TradingEnv(df_test_slice, feature_columns=feature_cols)
            rl_equity, rl_stats, rl_log = agent.evaluate(env_test)

            rl_metrics = RiskManager.compute_all_metrics(rl_equity)
            rl_metrics["equity"] = rl_equity
            rl_metrics["n_trades"] = rl_stats.get("total_trades", 0)
            rl_metrics["total_costs"] = rl_stats.get("total_costs", 0)
            rl_metrics["killed"] = rl_stats.get("killed", False)

            st.write(f"✅ RL trained — Return: {rl_metrics['total_return']:+.2%} | "
                     f"Sharpe: {rl_metrics['sharpe_ratio']:.3f}")
        except Exception as e:
            st.error(f"❌ RL training failed: {e}")
            st.stop()

        # ── Stage 5: Backtesting ──────────────────────────────────────────
        st.write("📊 **Stage 5/7** — Running backtests...")
        try:
            bt_engine = BacktestEngine()
            bt_results = bt_engine.run_ml_backtest(predictions, close_test, dates_test)

            benchmark_runner = BenchmarkRunner(df_test_slice)
            bm_results = benchmark_runner.run_all()

            all_metrics = {}
            for name, res in bt_results.items():
                all_metrics[name] = res
            all_metrics["PPO (RL)"] = rl_metrics
            for name, res in bm_results.items():
                all_metrics[f"BM: {name}"] = res

            st.write(f"✅ Backtested {len(bt_results)} ML strategies + {len(bm_results)} benchmarks")
        except Exception as e:
            st.error(f"❌ Backtesting failed: {e}")
            st.stop()

        # ── Stage 6: Monte Carlo ──────────────────────────────────────────
        mc_reports = {}
        if RUN_MONTE_CARLO:
            st.write("🎲 **Stage 6/7** — Monte Carlo simulation...")
            try:
                mc = MonteCarloSimulator(n_simulations=MC_SIMULATIONS)
                rl_returns = np.diff(rl_equity) / (rl_equity[:-1] + 1e-10)

                mc_block = mc.run_block_bootstrap(rl_returns, block_size=20)
                mc_block_report = mc.generate_report(mc_block)

                mc_stress = mc.run_stress_test(rl_returns)
                mc_stress_report = mc.generate_report(mc_stress)

                mc_perturb = mc.run_return_perturbation(rl_returns, noise_std=0.001)
                mc_perturb_report = mc.generate_report(mc_perturb)

                mc.plot_all(mc_block, save_dir=SAVE_DIR_MC)

                mc_reports = {
                    "block_bootstrap": mc_block_report,
                    "stress_test": mc_stress_report,
                    "perturbation": mc_perturb_report,
                }

                # Save MC CSV
                mc_rows = []
                for method, report in mc_reports.items():
                    row = {"method": method}
                    row.update(report)
                    mc_rows.append(row)
                mc_df = pd.DataFrame(mc_rows)
                mc_df.to_csv(os.path.join(SAVE_DIR_REPORTS, "monte_carlo_report.csv"), index=False)

                st.write(f"✅ Monte Carlo complete ({MC_SIMULATIONS:,} simulations × 3 methods)")
            except Exception as e:
                st.warning(f"⚠️ Monte Carlo failed: {e}")
        else:
            st.write("⏭️ **Stage 6/7** — Monte Carlo skipped")

        # ── Stage 7: Walk-Forward ─────────────────────────────────────────
        wf_results = pd.DataFrame()
        if RUN_WALK_FWD:
            st.write("📋 **Stage 7/7** — Walk-forward validation...")
            try:
                wf_results = walk_forward_validation(
                    df, feature_cols,
                    train_years=3, test_years=1, step_years=1,
                    embargo=EMBARGO_BARS,
                )
                st.write(f"✅ Walk-forward complete ({len(wf_results)} folds)")
            except Exception as e:
                st.warning(f"⚠️ Walk-forward failed: {e}")
        else:
            st.write("⏭️ **Stage 7/7** — Walk-forward skipped")

        # ── Generate Charts ───────────────────────────────────────────────
        st.write("📈 Generating charts...")
        try:
            plot_equity_curves(bt_results, rl_equity, bm_results, dates_test,
                               save_dir=SAVE_DIR_PLOTS)
            plot_prediction_charts(predictions, dates_test, close_test,
                                   save_dir=SAVE_DIR_PLOTS)
            plot_confusion_matrices(predictions, y_test, save_dir=SAVE_DIR_PLOTS)
            plot_accuracy_comparison(results_df, save_dir=SAVE_DIR_PLOTS)
            if not wf_results.empty:
                plot_walk_forward(wf_results, save_dir=SAVE_DIR_PLOTS)
            st.write("✅ All charts saved")
        except Exception as e:
            st.warning(f"⚠️ Chart generation error: {e}")

        # ── Save Reports ──────────────────────────────────────────────────
        try:
            save_risk_summary(all_metrics, save_dir=SAVE_DIR_REPORTS)
            save_comparison_csv(results_df, bt_results, save_dir=SAVE_DIR_REPORTS)
            if not wf_results.empty:
                wf_results.to_csv(
                    os.path.join(SAVE_DIR_REPORTS, "walk_forward_results.csv"), index=False
                )
        except Exception as e:
            st.warning(f"⚠️ Report save error: {e}")

        status.update(label="✅ Engine Complete!", state="complete", expanded=False)

    # ── Store results in session state ────────────────────────────────────
    st.session_state["engine_done"] = True
    st.session_state["results_df"] = results_df
    st.session_state["rl_metrics"] = rl_metrics
    st.session_state["rl_stats"] = rl_stats
    st.session_state["all_metrics"] = all_metrics
    st.session_state["predictions"] = predictions
    st.session_state["bt_results"] = bt_results
    st.session_state["bm_results"] = bm_results
    st.session_state["mc_reports"] = mc_reports
    st.session_state["wf_results"] = wf_results
    st.session_state["feature_cols"] = feature_cols
    st.session_state["max_acc"] = results_df["Accuracy"].max()
    st.session_state["df_shape"] = df.shape


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY (persisted via session_state)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("engine_done"):

    results_df    = st.session_state["results_df"]
    rl_metrics    = st.session_state["rl_metrics"]
    rl_stats      = st.session_state.get("rl_stats", {})
    all_metrics   = st.session_state["all_metrics"]
    mc_reports    = st.session_state.get("mc_reports", {})
    wf_results    = st.session_state.get("wf_results", pd.DataFrame())
    max_acc       = st.session_state["max_acc"]
    feature_cols  = st.session_state["feature_cols"]

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_overview, tab_equity, tab_ml, tab_mc, tab_wf = st.tabs([
        "📊 Overview",
        "📈 Equity Curves",
        "🔮 ML Predictions",
        "🎲 Monte Carlo",
        "📋 Walk-Forward",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    with tab_overview:

        # ── Leakage Check ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔒 Leakage Sanity Check</div>',
                    unsafe_allow_html=True)

        if max_acc <= 0.60:
            st.markdown(
                f'<span class="status-badge badge-clean">✅ CLEAN</span> '
                f'&nbsp; Max accuracy = {max_acc:.1%} — realistic range',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="status-badge badge-warning">⚠️ REVIEW NEEDED</span> '
                f'&nbsp; Max accuracy = {max_acc:.1%} — suspiciously high for daily prediction',
                unsafe_allow_html=True
            )

        # ── RL Metrics Cards ──────────────────────────────────────────────
        st.markdown('<div class="section-header">🧠 RL Agent Performance (PPO)</div>',
                    unsafe_allow_html=True)

        cols = st.columns(5)
        with cols[0]:
            render_metric_card("Total Return", rl_metrics["total_return"], fmt="pct")
        with cols[1]:
            render_metric_card("Sharpe Ratio", rl_metrics["sharpe_ratio"], fmt="float")
        with cols[2]:
            render_metric_card("Max Drawdown", rl_metrics["max_drawdown"], fmt="pct")
        with cols[3]:
            render_metric_card("Trades", rl_metrics.get("n_trades", 0), fmt="int")
        with cols[4]:
            killed = rl_metrics.get("killed", False)
            render_metric_card("Kill Switch", "TRIGGERED" if killed else "OK",
                               fmt="str")

        # Extra RL metrics row
        cols2 = st.columns(4)
        with cols2[0]:
            render_metric_card("Sortino Ratio", rl_metrics.get("sortino_ratio", 0), fmt="float")
        with cols2[1]:
            render_metric_card("Calmar Ratio", rl_metrics.get("calmar_ratio", 0), fmt="float")
        with cols2[2]:
            render_metric_card("Volatility (Ann.)", rl_metrics.get("volatility", 0), fmt="float")
        with cols2[3]:
            render_metric_card("Total Costs", rl_metrics.get("total_costs", 0), fmt="float",
                               prefix="$")

        # ── RL vs Benchmark ───────────────────────────────────────────────
        st.markdown('<div class="section-header">⚔️ RL vs Benchmarks</div>',
                    unsafe_allow_html=True)

        bm_results_state = st.session_state.get("bm_results", {})
        bh_sharpe = bm_results_state.get("Buy & Hold", {}).get("sharpe_ratio", 0)
        rl_sharpe = rl_metrics.get("sharpe_ratio", 0)

        if rl_sharpe > bh_sharpe:
            st.success(
                f"🏆 **RL WINS!** — RL Sharpe ({rl_sharpe:.3f}) > Buy&Hold ({bh_sharpe:.3f})"
            )
        else:
            st.warning(
                f"📉 **Needs more work** — RL Sharpe ({rl_sharpe:.3f}) ≤ Buy&Hold ({bh_sharpe:.3f})"
            )

        # ── ML Model Ranking ─────────────────────────────────────────────
        st.markdown('<div class="section-header">🏅 ML Model Ranking</div>',
                    unsafe_allow_html=True)

        styled_df = results_df.copy()
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy", format="%.4f", min_value=0, max_value=1
                ),
                "Precision": st.column_config.ProgressColumn(
                    "Precision", format="%.4f", min_value=0, max_value=1
                ),
                "Recall": st.column_config.ProgressColumn(
                    "Recall", format="%.4f", min_value=0, max_value=1
                ),
            },
        )

        # ── Config Summary ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📝 Config Summary</div>',
                    unsafe_allow_html=True)

        df_shape = st.session_state.get("df_shape", (0, 0))
        config_cols = st.columns(4)
        with config_cols[0]:
            render_metric_card("Features", len(feature_cols), fmt="int")
        with config_cols[1]:
            render_metric_card("Data Rows", df_shape[0], fmt="int")
        with config_cols[2]:
            render_metric_card("Embargo", EMBARGO_BARS, fmt="int", suffix=" bars")
        with config_cols[3]:
            render_metric_card("RL Timesteps", RL_TIMESTEPS, fmt="int")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: EQUITY CURVES
    # ══════════════════════════════════════════════════════════════════════
    with tab_equity:
        st.markdown('<div class="section-header">📈 Equity Curves — All Strategies vs Benchmarks</div>',
                    unsafe_allow_html=True)
        show_image_safe(
            os.path.join(SAVE_DIR_PLOTS, "equity_curves.png"),
            caption="Equity curves for all ML models, PPO RL agent, and benchmarks"
        )

        # ── Risk Summary Table ────────────────────────────────────────────
        risk_csv = os.path.join(SAVE_DIR_REPORTS, "risk_summary.csv")
        if os.path.exists(risk_csv):
            st.markdown('<div class="section-header">📋 Risk Summary</div>',
                        unsafe_allow_html=True)
            risk_df = pd.read_csv(risk_csv)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: ML PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════
    with tab_ml:
        # ── Accuracy Comparison ───────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Accuracy Comparison</div>',
                    unsafe_allow_html=True)
        show_image_safe(
            os.path.join(SAVE_DIR_PLOTS, "accuracy_comparison.png"),
            caption="Model accuracy comparison"
        )

        # ── Confusion Matrices ────────────────────────────────────────────
        st.markdown('<div class="section-header">🔢 Confusion Matrices</div>',
                    unsafe_allow_html=True)
        show_image_safe(
            os.path.join(SAVE_DIR_PLOTS, "confusion_matrices.png"),
            caption="Confusion matrices for all models"
        )

        # ── Individual Prediction Charts ──────────────────────────────────
        st.markdown('<div class="section-header">🔮 Per-Model Predictions</div>',
                    unsafe_allow_html=True)

        predictions_state = st.session_state.get("predictions", {})
        model_names = list(predictions_state.keys())

        if model_names:
            # 2-column layout for prediction charts
            for i in range(0, len(model_names), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(model_names):
                        name = model_names[idx]
                        safe_name = name.lower().replace(" ", "_")
                        path = os.path.join(SAVE_DIR_PLOTS, f"{safe_name}_prediction.png")
                        with col:
                            show_image_safe(path, caption=f"{name} — Test Period Prediction")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════
    with tab_mc:
        if mc_reports:
            st.markdown('<div class="section-header">🎲 Monte Carlo Validation</div>',
                        unsafe_allow_html=True)

            # ── MC Report Cards ───────────────────────────────────────────
            mc_tabs = st.tabs(["📦 Block Bootstrap", "💥 Stress Test", "🔀 Perturbation"])

            for mc_tab, (method, report) in zip(mc_tabs, mc_reports.items()):
                with mc_tab:
                    formatted = format_mc_report(report)
                    cols = st.columns(4)
                    items = list(formatted.items())
                    for k, (metric_name, metric_val) in enumerate(items):
                        with cols[k % 4]:
                            st.metric(label=metric_name, value=metric_val)

            # ── MC Plots ──────────────────────────────────────────────────
            st.markdown('<div class="section-header">📈 Monte Carlo Charts</div>',
                        unsafe_allow_html=True)

            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                show_image_safe(
                    os.path.join(SAVE_DIR_MC, "mc_equity_fan.png"),
                    caption="Equity Fan Chart"
                )
                show_image_safe(
                    os.path.join(SAVE_DIR_MC, "mc_return_hist.png"),
                    caption="Return Distribution"
                )
            with mc_col2:
                show_image_safe(
                    os.path.join(SAVE_DIR_MC, "mc_drawdown_hist.png"),
                    caption="Drawdown Distribution"
                )
                show_image_safe(
                    os.path.join(SAVE_DIR_MC, "mc_sharpe_hist.png"),
                    caption="Sharpe Ratio Distribution"
                )
        else:
            st.info("🎲 Monte Carlo simulation was not run. Enable it in the sidebar and re-run.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5: WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    with tab_wf:
        if not wf_results.empty:
            st.markdown('<div class="section-header">📋 Walk-Forward Validation</div>',
                        unsafe_allow_html=True)

            # ── Summary Metrics ───────────────────────────────────────────
            wf_cols = st.columns(4)
            with wf_cols[0]:
                render_metric_card("Folds", len(wf_results), fmt="int")
            with wf_cols[1]:
                render_metric_card("Avg Return",
                                   wf_results["total_return"].mean(), fmt="pct")
            with wf_cols[2]:
                avg_sharpe = wf_results["sharpe_ratio"].mean()
                render_metric_card("Avg Sharpe", avg_sharpe, fmt="float")
            with wf_cols[3]:
                std_sharpe = wf_results["sharpe_ratio"].std()
                render_metric_card("Sharpe Stability",
                                   "STABLE" if std_sharpe < 0.5 else "REVIEW",
                                   fmt="str")

            # ── Walk-Forward Plot ─────────────────────────────────────────
            show_image_safe(
                os.path.join(SAVE_DIR_PLOTS, "walk_forward.png"),
                caption="Walk-forward validation results per fold"
            )

            # ── Walk-Forward Table ────────────────────────────────────────
            st.markdown('<div class="section-header">📄 Fold Details</div>',
                        unsafe_allow_html=True)
            st.dataframe(wf_results, use_container_width=True, hide_index=True)
        else:
            st.info("📋 Walk-forward validation was not run. Enable it in the sidebar and re-run.")

else:
    # ── Welcome state ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #9CA3AF;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🥇</div>
        <h2 style="color: #D4A843; font-weight: 700; margin-bottom: 0.5rem;">
            Welcome to MLRL01
        </h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto; line-height: 1.6;">
            Configure your parameters in the sidebar, then hit
            <strong style="color: #D4A843;">Start Engine</strong>
            to begin the full quantitative trading pipeline.
        </p>
        <div style="
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        ">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🤖</div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.3rem;">7 ML Models</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.3rem;">PPO RL Agent</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.3rem;">Backtesting</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🎲</div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.3rem;">Monte Carlo</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">📋</div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.3rem;">Walk-Forward</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show existing results if available from previous runs
    if os.path.exists(os.path.join(SAVE_DIR_PLOTS, "equity_curves.png")):
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        with st.expander("📂 View Previous Run Results", expanded=False):
            st.caption("These are plots from your last CLI or Streamlit run:")
            show_image_safe(
                os.path.join(SAVE_DIR_PLOTS, "equity_curves.png"),
                caption="Equity Curves (previous run)"
            )
