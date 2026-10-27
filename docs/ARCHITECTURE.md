# System Architecture & Design Specification

## Overview

The **Laptop Power ML** application provides machine-learning-driven power consumption forecasting, battery runtime estimation, hardware telemetry logging, and actionable power-saving recommendations. The architecture enforces a clean separation of concerns across core modules.

```
laptop_power/
├── telemetry/          # Sensor discovery & cross-platform hardware readers (Windows CIM/WMI/ctypes, Linux sysfs)
├── data/               # Telemetry logging, session dataset management, outlier filtering, synthetic generator
├── features/           # Feature engineering, scaling, categorical encoding, schema validation
├── models/             # Candidate model training (Dummy, Ridge, RandomForest, HistGradientBoosting), evaluation, serialization
├── prediction/         # Inference engine (PowerPredictor) and EMA-smoothed runtime estimator (BatteryRuntimeEstimator)
├── recommendations/    # Data-driven power optimization advisory engine
├── dashboard/          # Streamlit dark-mode interactive web dashboard
```

## Layer Descriptions

### 1. Telemetry Layer (`laptop_power/telemetry/`)
- **`schema.py`**: Dataclass `TelemetrySample` providing a uniform data contract across operating systems.
- **`windows.py`**: Windows API interface using `ctypes` (`GetSystemPowerStatus`) and PowerShell/CIM queries (`Win32_Battery`, `root/wmi BatteryStatus`).
- **`linux.py`**: Linux sysfs reader inspecting `/sys/class/power_supply/BAT*` and `powerprofilesctl`.
- **`reader.py`**: Auto-detects operating system, queries system CPU/RAM/GPU metrics via `psutil`, handles missing sensors safely, and derives discharge power when voltage & current are exposed.

### 2. Data Management Layer (`laptop_power/data/`)
- **`collector.py`**: Background thread collector recording timestamped telemetry into session CSV files (`session_YYYYMMDD_HHMMSS.csv`).
- **`dataset.py`**: Loads session CSVs, drops duplicates, filters physically impossible outliers (e.g., negative power or draw >120W), and executes chronological/session-based train/test splits to avoid temporal data leakage.
- **`synthetic.py`**: Generates physics-guided synthetic telemetry datasets for demonstration mode, initial pipeline testing, and baseline model training.

### 3. Feature Engineering Layer (`laptop_power/features/`)
- Preprocesses numerical metrics using `StandardScaler` and categorical indicators using `OneHotEncoder`.
- Enforces strict input feature column schemas to prevent shape mismatches during inference.

### 4. Machine Learning & Models Layer (`laptop_power/models/`)
- **`trainer.py`**: Trains baseline and complex regressors (`DummyRegressor`, `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`).
- **`evaluator.py`**: Evaluates models on held-out test data using MAE (Watts), RMSE, R², and residual analysis.
- **`storage.py`**: Serializes fitted scikit-learn pipelines (`joblib`) alongside JSON metadata sidecars containing feature lists, metrics, and training timestamps.

### 5. Prediction & Runtime Estimation Layer (`laptop_power/prediction/`)
- **`predictor.py`**: Accepts live telemetry samples and outputs predicted power consumption in Watts with fallback handling.
- **`runtime_estimator.py`**: Calculates remaining battery life in hours and minutes (`Remaining Energy Wh / Smoothed Discharge W`) with Exponential Moving Average (EMA) smoothing and uncertainty indicators.

### 6. Power Recommendations Layer (`laptop_power/recommendations/`)
- Evaluates system state against advisory rules and historical workload session averages.
- Recommends reversible power-saving actions (switching power profiles, reducing display brightness, throttling CPU/GPU tasks) with quantified battery runtime extension estimates.

### 7. User Interface (`laptop_power/dashboard/`)
- Built with Streamlit, Plotly, and custom CSS glassmorphism styling.
- Renders live metrics, prediction timelines, gauge meters, training matrices, and data management controls.
