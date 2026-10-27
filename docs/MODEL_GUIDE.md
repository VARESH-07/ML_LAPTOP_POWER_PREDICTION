# Machine Learning Model & Evaluation Guide

## Target & Features

- **Target Variable**: `discharge_power_w` (Instantaneous battery discharge power in Watts).
- **Numerical Features**: `cpu_util_pct`, `cpu_freq_mhz`, `ram_util_pct`, `ram_used_gb`, `gpu_util_pct`, `battery_pct`, `battery_voltage_v`.
- **Categorical Features**: `charging_status`, `workload_label`, `power_profile`.

## Candidate Models

1. **DummyRegressor (Baseline)**: Predicts mean training target value. Serves as reference point for minimum acceptable performance.
2. **Ridge Regression**: Linear regressor with L2 regularization. Evaluates linear relationship between utilization and power draw.
3. **RandomForestRegressor**: Ensemble of decision trees. Captures non-linear interactions between CPU, GPU, RAM, and power profiles.
4. **HistGradientBoostingRegressor**: Fast gradient boosted decision trees. Excellent for tabular telemetry data.

## Evaluation Metrics & Validation Strategy

To prevent data leakage, training datasets are split chronologically or session-by-session. Metrics reported on held-out test data:

- **MAE (Mean Absolute Error)**: Average magnitude of prediction error in Watts.
- **RMSE (Root Mean Squared Error)**: Penalizes large error outliers.
- **R² Score**: Proportion of variance in power draw explained by telemetry features.
- **Max Error**: Maximum single prediction deviation in Watts.

## Preprocessing & Artifact Storage

Saved model pipelines and metadata are stored in `saved_models/`:
- `power_model_latest.joblib`: Complete scikit-learn Pipeline object containing fitted scaler, encoder, and regressor.
- `power_model_latest_meta.json`: JSON metadata sidecar specifying model class, training timestamp, feature list, evaluation metrics, and sample count.
