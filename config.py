"""
Global Configuration for Laptop Power ML Application.
Contains hardware defaults, sensor sampling configurations, feature names, and paths.
"""

from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_store"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "saved_models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Fallbacks (Target: ASUS Vivobook 14 M1407KA / AMD Ryzen AI 7 350)
DESIGN_BATTERY_CAPACITY_WH = 42.0  # Wh design capacity
DESIGN_VOLTAGE_V = 11.486           # Nominal voltage V

# Telemetry Sampling Settings
DEFAULT_SAMPLE_INTERVAL_SEC = 5
MIN_SAMPLE_INTERVAL_SEC = 1
MAX_SAMPLE_INTERVAL_SEC = 60

# Model Feature Definitions
NUMERICAL_FEATURES = [
    "cpu_util_pct",
    "cpu_freq_mhz",
    "ram_util_pct",
    "ram_used_gb",
    "gpu_util_pct",
    "battery_pct",
    "battery_voltage_v",
]

CATEGORICAL_FEATURES = [
    "charging_status",  # 'Discharging', 'Charging', 'Full', 'Unknown'
    "workload_label",    # 'Idle', 'Browsing', 'Coding', 'Compiling', 'Video Playback', 'Gaming'
    "power_profile",     # 'power-saver', 'balanced', 'performance', 'unknown'
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "discharge_power_w"
RUNTIME_TARGET_COLUMN = "remaining_runtime_min"

# Workload Presets for Synthetic/Demo Generation & Simulation
WORKLOAD_PRESETS = {
    "Idle": {"cpu_range": (2.0, 10.0), "ram_gb_range": (3.5, 5.0), "gpu_range": (0.0, 5.0), "base_power_w": 4.5},
    "Browsing": {"cpu_range": (10.0, 25.0), "ram_gb_range": (5.0, 7.5), "gpu_range": (5.0, 15.0), "base_power_w": 9.0},
    "Coding": {"cpu_range": (15.0, 35.0), "ram_gb_range": (6.0, 9.0), "gpu_range": (2.0, 10.0), "base_power_w": 11.0},
    "Compiling": {"cpu_range": (60.0, 95.0), "ram_gb_range": (8.0, 13.0), "gpu_range": (5.0, 20.0), "base_power_w": 28.0},
    "Video Playback": {"cpu_range": (12.0, 28.0), "ram_gb_range": (5.5, 8.0), "gpu_range": (20.0, 45.0), "base_power_w": 14.0},
    "Gaming": {"cpu_range": (50.0, 90.0), "ram_gb_range": (9.0, 14.5), "gpu_range": (70.0, 98.0), "base_power_w": 36.0},
}
