"""
Laptop Power ML — Main CLI Entrypoint & Runner.
Provides CLI flags to sample telemetry, train models, collect session data, or launch dashboard.
"""

import argparse
import sys
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from laptop_power.telemetry import TelemetryReader
from laptop_power.prediction import PowerPredictor, BatteryRuntimeEstimator
from laptop_power.recommendations import RecommendationEngine
from laptop_power.data import DataCollector, DatasetManager
from laptop_power.models import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_sample():
    reader = TelemetryReader()
    sample = reader.read_sample(workload_label="CLI Sample")
    predictor = PowerPredictor()
    pred = predictor.predict_sample(sample)
    runtime_est = BatteryRuntimeEstimator()
    runtime_res = runtime_est.estimate_runtime(sample, predicted_power_w=pred["predicted_power_w"])
    rec_engine = RecommendationEngine()
    recs = rec_engine.generate_recommendations(sample)

    print("\n" + "=" * 60)
    print(" LIVE TELEMETRY & PREDICTION READOUT ")
    print("=" * 60)
    print(f"OS Environment:      {sample.os_name}")
    print(f"Battery Level:       {sample.battery_pct:.1f}% ({sample.charging_status})")
    print(f"Battery Voltage:     {sample.battery_voltage_v or 'N/A'} V")
    print(f"Discharge Draw:      {sample.discharge_power_w or '0.0 (Charging/Plugged in)'} W")
    print(f"CPU Utilization:     {sample.cpu_util_pct:.1f}% ({sample.cpu_freq_mhz:.0f} MHz)")
    print(f"RAM Utilization:     {sample.ram_util_pct:.1f}% ({sample.ram_used_gb:.1f} GB)")
    print(f"GPU Utilization:     {sample.gpu_util_pct:.1f}%")
    print("-" * 60)
    print(f"ML Predicted Power:  {pred['predicted_power_w']:.2f} W ({pred['model_name']})")
    print(f"Estimated Runtime:   {runtime_res['runtime_formatted']} (Confidence: {runtime_res['confidence']})")
    if runtime_res.get("warning"):
        print(f"Runtime Warning:     {runtime_res['warning']}")
    print("-" * 60)
    print(f"Active Recommendations ({len(recs)}):")
    for r in recs:
        print(f"  • [{r.category}] {r.title} (Est. -{r.potential_savings_w}W, +{r.estimated_battery_extension_min}m)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Laptop Power ML Application")
    parser.add_argument("--sample", action="store_true", help="Read a single live telemetry sample & prediction")
    parser.add_argument("--train", action="store_true", help="Train and evaluate ML candidate models")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit Web Dashboard")

    args = parser.parse_args()

    if args.sample:
        run_sample()
    elif args.train:
        trainer = ModelTrainer()
        trainer.train_and_evaluate_all()
    elif args.dashboard:
        import subprocess
        app_path = BASE_DIR / "laptop_power" / "dashboard" / "app.py"
        python_exe = sys.executable
        subprocess.run([python_exe, "-m", "streamlit", "run", str(app_path)])
    else:
        # Default behavior: run live sample
        run_sample()


if __name__ == "__main__":
    main()
