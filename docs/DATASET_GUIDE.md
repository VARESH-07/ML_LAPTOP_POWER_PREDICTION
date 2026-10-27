# Dataset Format & Data Collection Guide

## Overview

The application logs real-time hardware telemetry into CSV files stored in `data_store/raw/`. Each session file is named `session_YYYYMMDD_HHMMSS.csv`.

## Telemetry Schema & Features

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | Float | Epoch timestamp (seconds) | `1791563882.85` |
| `datetime_str` | String | ISO formatted timestamp | `2026-10-09T22:08:02` |
| `os_name` | String | Operating System ('Windows', 'Linux') | `Windows` |
| `battery_pct` | Float | Battery state of charge (%) | `64.0` |
| `charging_status` | String | Battery state ('Discharging', 'Charging', 'Full') | `Discharging` |
| `battery_voltage_v` | Float | Battery terminal voltage (Volts) | `11.075` |
| `battery_current_a` | Float | Discharge current (Amperes) | `0.917` |
| `battery_power_w` | Float | Battery power draw (Watts) | `10.156` |
| `discharge_power_w` | Float | **Target Column**: Discharge power in Watts | `10.156` |
| `energy_remaining_wh`| Float | Usable battery energy remaining (Wh) | `26.88` |
| `energy_full_wh` | Float | Design/Full charge capacity (Wh) | `42.0` |
| `cpu_util_pct` | Float | Overall CPU utilization (%) | `15.9` |
| `cpu_freq_mhz` | Float | Primary CPU clock frequency (MHz) | `2000.0` |
| `ram_util_pct` | Float | RAM utilization (%) | `85.8` |
| `ram_used_gb` | Float | RAM used (GB) | `12.97` |
| `gpu_util_pct` | Float | GPU utilization (%) | `2.5` |
| `power_profile` | String | Active profile ('balanced', 'power-saver') | `balanced` |
| `workload_label` | String | Workload preset tag ('Idle', 'Coding', 'Gaming') | `Coding` |
| `is_synthetic` | Bool | Flag indicating real vs synthetic sample | `False` |
| `session_id` | String | Unique collection session identifier | `session_20261009_220800` |

## How to Collect Real Training Data

1. Launch the application or dashboard:
   ```bash
   python main.py --dashboard
   ```
2. Set the **Active Workload Tag** in the sidebar (e.g., `Idle`, `Browsing`, `Coding`, `Compiling`, `Video Playback`, `Gaming`).
3. Click **▶ Start Logging**.
4. Use your laptop under that workload for 15-30 minutes.
5. Switch workload tags and repeat for comprehensive coverage across different power states.
6. Click **⏹ Stop Logging** when finished.
7. Navigate to **Model Training & Evaluation** tab in the dashboard or run `python train.py` to retrain models on your newly collected real data.
