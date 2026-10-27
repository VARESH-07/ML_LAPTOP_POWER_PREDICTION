"""
Laptop Power ML — Streamlit Interactive Web Dashboard.
Visualizes live battery telemetry, ML power predictions, remaining battery runtime,
power optimization recommendations, and model training metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from laptop_power.telemetry import TelemetryReader, TelemetrySample
from laptop_power.data import DataCollector, DatasetManager, generate_synthetic_dataset
from laptop_power.models import ModelTrainer, ModelStorage, ModelEvaluator
from laptop_power.prediction import PowerPredictor, BatteryRuntimeEstimator
from laptop_power.recommendations import RecommendationEngine

logger = logging.getLogger(__name__)

# Set Streamlit Page Config
st.set_page_config(
    page_title="Laptop Power ML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Cyberpunk / Modern Glassmorphism Styling
CUSTOM_CSS = """
<style>
    /* Dark theme background overrides */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Card Container Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Status Badge Styling */
    .badge-discharging {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-charging {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-high {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }

    .badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }

    .rec-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_components():
    """Initializes singleton components across Streamlit re-runs."""
    reader = TelemetryReader()
    collector = DataCollector()
    dataset_mgr = DatasetManager()
    storage = ModelStorage()
    predictor = PowerPredictor(storage=storage)
    runtime_est = BatteryRuntimeEstimator()
    rec_engine = RecommendationEngine()
    return reader, collector, dataset_mgr, storage, predictor, runtime_est, rec_engine


def main():
    reader, collector, dataset_mgr, storage, predictor, runtime_est, rec_engine = get_components()

    # Session State Initialization for Live Streaming History
    if "history" not in st.session_state:
        st.session_state.history = []
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False

    # Header / Title Bar
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.title("⚡ Laptop Power ML Engine")
        st.caption("Target System: ASUS Vivobook 14 (AMD Ryzen AI 7 350 | Radeon 860M | 42 Wh Battery)")
    with col_status:
        st.write("")
        st.write("")
        st.session_state.demo_mode = st.toggle("Demo / Simulation Mode", value=st.session_state.demo_mode)

    # Sidebar Navigation & Controls
    st.sidebar.image("https://img.icons8.com/isometric/100/lightning-bolt.png", width=60)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Live Telemetry & Prediction", "Model Training & Evaluation", "Data Management", "Documentation"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telemetry Session Logging")
    workload_input = st.sidebar.selectbox("Active Workload Tag", list(config.WORKLOAD_PRESETS.keys()), index=0)
    interval_input = st.sidebar.slider("Sample Interval (s)", 2, 30, config.DEFAULT_SAMPLE_INTERVAL_SEC)
    collector.interval_sec = interval_input

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶ Start Logging", disabled=collector.is_running()):
            sid = collector.start_session(workload_label=workload_input)
            st.sidebar.success(f"Logging: {sid}")
    with col_btn2:
        if st.button("⏹ Stop Logging", disabled=not collector.is_running()):
            collector.stop_session()
            st.sidebar.info("Logging Stopped")

    if collector.is_running():
        st.sidebar.warning(f"🔴 Live Session Active: {collector.current_session_id}")

    # ==========================================
    # TAB 1: LIVE TELEMETRY & PREDICTION DASHBOARD
    # ==========================================
    if app_mode == "Live Telemetry & Prediction":
        # Sample current telemetry
        if st.session_state.demo_mode:
            sample = TelemetrySample(
                timestamp=time.time(),
                datetime_str=pd.to_datetime("now").isoformat(),
                os_name="Windows (Simulated)",
                battery_pct=72.5,
                charging_status="Discharging",
                battery_voltage_v=11.2,
                battery_current_a=0.95,
                battery_power_w=10.64,
                discharge_power_w=10.64,
                energy_remaining_wh=30.45,
                energy_full_wh=42.0,
                cpu_util_pct=24.5,
                cpu_freq_mhz=2400.0,
                ram_util_pct=48.2,
                ram_used_gb=7.7,
                gpu_util_pct=12.0,
                power_profile="balanced",
                workload_label=workload_input,
                is_synthetic=True
            )
        else:
            sample = reader.read_sample(workload_label=workload_input)

        # ML Prediction & Battery Runtime Calculation
        pred_res = predictor.predict_sample(sample)
        predicted_w = pred_res["predicted_power_w"]
        runtime_res = runtime_est.estimate_runtime(sample, predicted_power_w=predicted_w)

        # Append to live history deque
        st.session_state.history.append({
            "timestamp": sample.timestamp,
            "datetime": sample.datetime_str,
            "discharge_power_w": sample.discharge_power_w or 0.0,
            "predicted_power_w": predicted_w,
            "cpu_util_pct": sample.cpu_util_pct,
            "ram_util_pct": sample.ram_util_pct,
            "gpu_util_pct": sample.gpu_util_pct,
            "battery_pct": sample.battery_pct,
        })
        if len(st.session_state.history) > 60:
            st.session_state.history.pop(0)

        # Overview Metrics Row
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            badge_cls = "badge-charging" if sample.charging_status == "Charging" else "badge-discharging"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Battery Level</div>
                <div class="metric-value">{sample.battery_pct:.1f}% <span class="{badge_cls}">{sample.charging_status}</span></div>
                <div class="metric-sub">Voltage: {sample.battery_voltage_v or 'N/A'} V | Cap: {sample.energy_remaining_wh or 'N/A'} Wh</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            obs_power_str = f"{sample.discharge_power_w:.2f} W" if sample.discharge_power_w is not None else "Plugged in / Unknown"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Measured Discharge Draw</div>
                <div class="metric-value" style="color: #f59e0b;">{obs_power_str}</div>
                <div class="metric-sub">Hardware Reported / Calculated</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            mae_tag = f"(MAE ±{pred_res['model_mae']:.2f}W)" if pred_res['model_mae'] else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Power Consumption</div>
                <div class="metric-value" style="color: #a855f7;">{predicted_w:.2f} W</div>
                <div class="metric-sub">{pred_res['model_name']} {mae_tag}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            conf_badge = f"badge-{runtime_res['confidence'].lower()}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Est. Remaining Runtime</div>
                <div class="metric-value" style="color: #10b981;">{runtime_res['runtime_formatted']}</div>
                <div class="metric-sub">Confidence: <span class="{conf_badge}">{runtime_res['confidence']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        if runtime_res.get("warning"):
            st.warning(f"⚠️ {runtime_res['warning']}")

        st.markdown("---")

        # Live Charts & Hardware Telemetry Gauges
        c_chart, c_gauges = st.columns([2, 1])

        with c_chart:
            st.subheader("📈 Live Power Consumption Timeline (Observed vs ML Prediction)")
            if len(st.session_state.history) > 0:
                hist_df = pd.DataFrame(st.session_state.history)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_df["datetime"], y=hist_df["discharge_power_w"],
                    mode="lines+markers", name="Observed Discharge (W)",
                    line=dict(color="#f59e0b", width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=hist_df["datetime"], y=hist_df["predicted_power_w"],
                    mode="lines+markers", name="ML Model Predicted (W)",
                    line=dict(color="#a855f7", width=2, dash="dash")
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

        with c_gauges:
            st.subheader("💻 Telemetry Gauges")
            st.write(f"**CPU Utilization:** `{sample.cpu_util_pct:.1f}%` ({sample.cpu_freq_mhz:.0f} MHz)")
            st.progress(min(1.0, sample.cpu_util_pct / 100.0))

            st.write(f"**RAM Utilization:** `{sample.ram_util_pct:.1f}%` ({sample.ram_used_gb:.1f} GB / 16GB)")
            st.progress(min(1.0, sample.ram_util_pct / 100.0))

            st.write(f"**GPU Utilization:** `{sample.gpu_util_pct:.1f}%`")
            st.progress(min(1.0, sample.gpu_util_pct / 100.0))

            st.write(f"**Active Power Profile:** `{sample.power_profile}`")
            st.write(f"**Workload Tag:** `{sample.workload_label}`")

        st.markdown("---")

        # Power Optimization Recommendations Panel
        st.subheader("💡 Data-Driven Power Optimization Advisory")
        hist_data = pd.DataFrame(st.session_state.history) if len(st.session_state.history) > 5 else None
        recs = rec_engine.generate_recommendations(sample, recent_history_df=hist_data)

        if recs:
            for r in recs:
                cmd_box = f"<code>{r.command_hint}</code>" if r.command_hint else ""
                ext_str = f"🔋 Est. +{r.estimated_battery_extension_min} mins runtime extension" if r.estimated_battery_extension_min > 0 else ""
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:1.05rem; color:#f8fafc;">[{r.category}] {r.title}</span>
                        <span style="color:#38bdf8; font-weight:600;">-{r.potential_savings_w:.1f} W</span>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.9rem; margin-top:4px;">{r.description}</div>
                    <div style="margin-top:6px; font-size:0.85rem; color:#34d399; font-weight:600;">{ext_str} {cmd_box}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("System is operating in optimal power-saving state. No high-power advisory warnings active.")

        # Auto-refresh loop
        time.sleep(2)
        st.rerun()

    # ==========================================
    # TAB 2: MODEL TRAINING & EVALUATION
    # ==========================================
    elif app_mode == "Model Training & Evaluation":
        st.header("🧠 Machine Learning Model Training & Validation")

        st.markdown("""
        Compare candidate supervised regression algorithms on recorded/synthetic telemetry datasets.
        Evaluates models using **MAE (Watts)**, **RMSE**, **R²**, and **Residual Analysis** on held-out test sessions.
        """)

        combined_df = dataset_mgr.load_combined_dataset(include_synthetic_fallback=True)
        is_syn = combined_df.get("is_synthetic", pd.Series([False])).iloc[0] if not combined_df.empty else True

        if is_syn:
            st.warning("⚠️ Currently evaluating on **Synthetic Telemetry Baseline Data**. Collect real hardware sessions for exact system calibration.")
        else:
            st.success(f"✅ Loaded {len(combined_df)} real hardware telemetry samples across recorded sessions.")

        if st.button("🚀 Train & Compare Candidate Models"):
            with st.spinner("Training Dummy, Ridge, RandomForest, and HistGradientBoosting regressors..."):
                trainer = ModelTrainer(data_manager=dataset_mgr, storage=storage)
                best_pipe, best_metrics, comparison_df = trainer.train_and_evaluate_all(combined_df)
                st.session_state["comparison_df"] = comparison_df
                st.session_state["best_metrics"] = best_metrics
                st.success(f"Model Training Complete! Best algorithm selected with MAE: {best_metrics['mae']} W")

        # Load existing model metadata
        pipeline, metadata = storage.load_model()

        if metadata:
            st.subheader("📌 Currently Deployed Model Metadata")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Selected Model", metadata.get("model_name"))
            mc2.metric("MAE (Watts)", f"{metadata.get('metrics', {}).get('mae')} W")
            mc3.metric("R² Score", f"{metadata.get('metrics', {}).get('r2')}")
            mc4.metric("Training Samples", metadata.get("num_training_samples"))

        if "comparison_df" in st.session_state:
            st.subheader("📊 Candidate Model Comparison Matrix")
            st.dataframe(st.session_state["comparison_df"], use_container_width=True)

            fig_comp = px.bar(
                st.session_state["comparison_df"],
                x="Model",
                y="MAE (W)",
                color="R² Score",
                title="Model Mean Absolute Error Comparison (Lower is Better)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ==========================================
    # TAB 3: DATA MANAGEMENT
    # ==========================================
    elif app_mode == "Data Management":
        st.header("📁 Dataset Management & Telemetry Storage")

        csv_files = list(config.RAW_DATA_DIR.glob("*.csv"))
        st.write(f"**Data Store Directory:** `{config.RAW_DATA_DIR}`")
        st.write(f"**Total Session Files:** `{len(csv_files)}`")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("⚡ Generate Synthetic Dataset Baseline (1200 samples)"):
                syn_path = config.RAW_DATA_DIR / "synthetic_demo.csv"
                generate_synthetic_dataset(num_samples=1200, output_path=syn_path)
                st.success("Generated synthetic demo dataset!")
                st.rerun()

        with col_d2:
            if st.button("🗑️ Clear All Local Session Datasets"):
                for f in csv_files:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                st.info("Cleared local session CSV datasets.")
                st.rerun()

        combined_df = dataset_mgr.load_combined_dataset(include_synthetic_fallback=False)
        if not combined_df.empty:
            st.subheader("Raw Recorded Telemetry Preview")
            st.dataframe(combined_df.head(100), use_container_width=True)
            st.subheader("Summary Statistics")
            st.dataframe(combined_df.describe(), use_container_width=True)
        else:
            st.info("No recorded datasets found in data store.")

    # ==========================================
    # TAB 4: DOCUMENTATION & SYSTEM SPECS
    # ==========================================
    elif app_mode == "Documentation":
        st.header("📖 Architecture & Hardware Documentation")
        st.markdown("""
        ### Target Hardware Specs
        - **Model:** ASUS Vivobook 14 M1407KA
        - **CPU:** AMD Ryzen AI 7 350
        - **GPU:** AMD Radeon 860M Integrated Graphics
        - **RAM:** 16 GB DDR5
        - **Battery Capacity:** ~42 Wh

        ### ML Architecture Overview
        1. **Telemetry Layer:** Cross-platform fallback reader supporting Windows WMI/CIM/ctypes and Linux `/sys/class/power_supply/`.
        2. **Preprocessing Pipeline:** Scikit-learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
        3. **Supervised Regressor:** Candidates evaluated via session-split MAE, RMSE, and R². Best estimator stored as joblib pipeline artifact.
        4. **Battery Runtime Engine:** Physics-based usable energy division smoothed via Exponential Moving Average (EMA).
        5. **Advisory Engine:** Rules and historical session contrast for non-disruptive power recommendations.
        """)


if __name__ == "__main__":
    main()
