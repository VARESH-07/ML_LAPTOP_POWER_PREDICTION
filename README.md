# ⚡ LAPTOP POWER — ML-BASED POWER CONSUMPTION & BATTERY LIFE PREDICTION

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![ML: scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning application that analyzes live laptop operating conditions, predicts instantaneous power consumption in Watts, estimates remaining battery runtime in hours and minutes, logs telemetry datasets, and provides data-driven power-saving optimization recommendations.

---

## 🎯 Target Hardware & System Specifications

- **Primary Target**: ASUS Vivobook 14 M1407KA
- **CPU**: AMD Ryzen AI 7 350
- **GPU**: AMD Radeon 860M Integrated Graphics
- **RAM**: 16 GB DDR5
- **Battery Capacity**: ~42 Wh (Design Capacity)
- **Supported OS**: Windows 10/11 & Linux (Ubuntu, Debian, Fedora, Arch)

---

## 🏗️ Software Architecture

The application is structured into modular decoupled packages:

```
MACHINE LEARNING/
├── config.py                 # Global paths, features, targets, hardware fallbacks
├── main.py                   # Main CLI entrypoint (telemetry readout, model training, UI launcher)
├── train.py                  # Standalone CLI model training & candidate comparison script
├── requirements.txt          # Python dependencies
├── laptop_power/
│   ├── telemetry/            # Sensor discovery, Windows WMI/CIM/ctypes & Linux sysfs readers
│   ├── data/                 # Data collection, session logger, outlier cleaner, synthetic generator
│   ├── features/             # Preprocessing pipelines, scaling, encoding, schema validation
│   ├── models/               # Model comparison (Dummy, Ridge, RandomForest, HistGradientBoosting)
│   ├── prediction/           # PowerPredictor inference & EMA-smoothed BatteryRuntimeEstimator
│   ├── recommendations/      # Advisory engine for profile, CPU, display, & session optimization
│   └── dashboard/            # Streamlit interactive web dashboard (dark theme, glassmorphism)
├── tests/                    # Automated pytest test suite
└── docs/                     # Architecture, dataset, model, troubleshooting & OS guides
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### Linux (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Read Live Telemetry & Prediction (CLI)

Run a quick live telemetry readout:
```bash
python main.py --sample
```

**Sample Output**:
```
============================================================
 LIVE TELEMETRY & PREDICTION READOUT 
============================================================
OS Environment:      Windows
Battery Level:       50.0% (Discharging)
Battery Voltage:     11.021 V
Discharge Draw:      12.244 W
CPU Utilization:     14.6% (2000 MHz)
RAM Utilization:     85.7% (13.0 GB)
GPU Utilization:     2.9%
------------------------------------------------------------
ML Predicted Power:  24.69 W (Random Forest Regressor)
Estimated Runtime:   1h 42m (Confidence: Medium)
------------------------------------------------------------
Active Recommendations (2):
  • [Power Profile] Switch to Power Saver Profile (Est. -1.8W, +17m)
  • [Display/GPU] Reduce Screen Brightness (Est. -2.0W, +20m)
============================================================
```

---

### 3. Train & Compare Candidate Models

Run candidate model training and cross-validation on session datasets:
```bash
python train.py
```

**Training Results Example**:
```
============================================================
 CANDIDATE MODEL EVALUATION COMPARISON 
============================================================
                  Model  MAE (W)  RMSE (W)  R² Score  Max Error (W)
Random Forest Regressor   0.5320    1.2189    0.9924        15.9668
   HistGradientBoosting   0.6282    1.0215    0.9946         5.9842
       Ridge Regression   1.0333    1.3775    0.9902         5.5241
DummyRegressor Baseline  12.1608   13.9589   -0.0019        22.0390
============================================================
BEST MODEL DEPLOYED: Random Forest Regressor
HELD-OUT MAE: 0.5320 Watts | HELD-OUT R² SCORE: 0.9924
============================================================
```

---

### 4. Launch Interactive Web Dashboard

Launch the Streamlit dark-mode dashboard:
```bash
python main.py --dashboard
```
*(Or directly: `streamlit run laptop_power/dashboard/app.py`)*

---

### 5. Execute Automated Test Suite

Run unit and integration tests:
```bash
python -m pytest -v
```

---

## 📚 Documentation Index

- [System Architecture](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/ARCHITECTURE.md)
- [Dataset Format & Collection Guide](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/DATASET_GUIDE.md)
- [Model Training & Evaluation Guide](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/MODEL_GUIDE.md)
- [Troubleshooting & Operational Guide](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/TROUBLESHOOTING.md)
- [Linux Setup Guide](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/LINUX_GUIDE.md)
- [Windows Compatibility Guide](file:///c:/Users/S%20RAJA%20VARESHWAR/Documents/MACHINE%20LEARNING/docs/WINDOWS_GUIDE.md)

---

## ⚠️ Model Limitations & Measurement Uncertainty

1. **Hardware Power Sensors**: Component-level power draw (e.g. CPU package power vs GPU core power) requires vendor-specific kernel drivers. When component power rails are unexposed, total system battery discharge power (Watts) is measured at the battery terminal.
2. **Charging State**: Discharge power measurement is non-zero only while operating on battery power (Discharging state). When plugged into AC power, remaining runtime calculations switch to "Plugged in (Charging/Full)".
3. **Transient Workload Smoothing**: Remaining runtime is estimated using Exponential Moving Average (EMA) smoothed discharge power to prevent erratic fluctuations during temporary CPU spikes.
