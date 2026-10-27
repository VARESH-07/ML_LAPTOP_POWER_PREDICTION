"""
Synthetic Telemetry & Power Data Generator.
Generates physics-guided synthetic datasets for initial pipeline tests, model baseline training,
and UI demonstration mode.
"""

import numpy as np
import pandas as pd
import time
import random
from typing import Optional
from pathlib import Path

import config


def generate_synthetic_dataset(
    num_samples: int = 1200,
    sample_interval_sec: int = 5,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic telemetry dataset representing typical laptop usage sessions.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    start_time = time.time() - (num_samples * sample_interval_sec)
    current_battery_pct = 95.0

    workload_names = list(config.WORKLOAD_PRESETS.keys())
    current_workload = "Idle"
    workload_duration = 0

    profiles = ["power-saver", "balanced", "performance"]

    for i in range(num_samples):
        timestamp = start_time + (i * sample_interval_sec)
        datetime_str = pd.to_datetime(timestamp, unit='s').isoformat()

        # Switch workload periodically
        if workload_duration <= 0:
            current_workload = random.choice(workload_names)
            workload_duration = random.randint(20, 60) # 20-60 samples per workload session
        workload_duration -= 1

        preset = config.WORKLOAD_PRESETS[current_workload]
        cpu_util = float(np.clip(np.random.normal(
            loc=(preset["cpu_range"][0] + preset["cpu_range"][1]) / 2.0,
            scale=(preset["cpu_range"][1] - preset["cpu_range"][0]) / 4.0
        ), 1.0, 100.0))

        ram_used_gb = float(np.clip(np.random.normal(
            loc=(preset["ram_gb_range"][0] + preset["ram_gb_range"][1]) / 2.0,
            scale=0.5
        ), 2.0, 15.5))
        ram_util = float(np.clip((ram_used_gb / 16.0) * 100.0, 10.0, 98.0))

        gpu_util = float(np.clip(np.random.normal(
            loc=(preset["gpu_range"][0] + preset["gpu_range"][1]) / 2.0,
            scale=5.0
        ), 0.0, 100.0))

        cpu_freq = float(np.clip(1600.0 + (cpu_util * 25.0) + np.random.normal(0, 100), 1200.0, 4300.0))

        profile = random.choice(profiles)
        profile_multiplier = 0.85 if profile == "power-saver" else (1.20 if profile == "performance" else 1.0)

        # Physics-guided power calculation:
        # P_total = (P_idle + P_cpu(util) + P_gpu(util) + P_ram(gb)) * profile_factor + noise
        base_idle = preset["base_power_w"]
        cpu_power = (cpu_util / 100.0) * 18.0   # Up to 18W CPU power
        gpu_power = (gpu_util / 100.0) * 12.0   # Up to 12W GPU power
        ram_power = (ram_used_gb / 16.0) * 2.5   # Up to 2.5W RAM power

        total_power = (base_idle + cpu_power + gpu_power + ram_power) * profile_multiplier
        noise = np.random.normal(0, 0.4)
        discharge_power_w = float(np.clip(total_power + noise, 3.5, 45.0))

        # Battery state logic (90% discharging sessions, 10% charging)
        charging_status = "Discharging" if (i % 200 != 0 and current_battery_pct > 10.0) else "Charging"
        if charging_status == "Discharging":
            # Drain energy based on power draw over interval
            wh_drained = (discharge_power_w * sample_interval_sec) / 3600.0
            pct_drained = (wh_drained / config.DESIGN_BATTERY_CAPACITY_WH) * 100.0
            current_battery_pct = max(5.0, current_battery_pct - pct_drained)
        else:
            current_battery_pct = min(100.0, current_battery_pct + 0.2)
            discharge_power_w = 0.0

        voltage_v = float(np.clip(11.4 - (1.0 - current_battery_pct / 100.0) * 1.2 + np.random.normal(0, 0.05), 9.8, 12.6))
        energy_rem_wh = (current_battery_pct / 100.0) * config.DESIGN_BATTERY_CAPACITY_WH

        records.append({
            "timestamp": timestamp,
            "datetime_str": datetime_str,
            "os_name": "Windows",
            "battery_pct": round(current_battery_pct, 2),
            "charging_status": charging_status,
            "battery_voltage_v": round(voltage_v, 3),
            "battery_current_a": round(discharge_power_w / voltage_v, 3) if discharge_power_w > 0 else 0.0,
            "battery_power_w": round(discharge_power_w, 3),
            "discharge_power_w": round(discharge_power_w, 3),
            "energy_remaining_wh": round(energy_rem_wh, 3),
            "energy_full_wh": config.DESIGN_BATTERY_CAPACITY_WH,
            "cpu_util_pct": round(cpu_util, 2),
            "cpu_freq_mhz": round(cpu_freq, 1),
            "ram_util_pct": round(ram_util, 2),
            "ram_used_gb": round(ram_used_gb, 2),
            "gpu_util_pct": round(gpu_util, 2),
            "power_profile": profile,
            "workload_label": current_workload,
            "is_synthetic": True,
            "session_id": "syn_session_01"
        })

    df = pd.DataFrame(records)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
