"""
Prediction and Battery Runtime Unit Tests.
"""
import pytest
from laptop_power.telemetry import TelemetrySample
from laptop_power.prediction import PowerPredictor, BatteryRuntimeEstimator


def test_power_predictor(tmp_path):
    predictor = PowerPredictor()

    sample = TelemetrySample(
        timestamp=1000.0,
        datetime_str="2026-10-09T00:00:00",
        os_name="Windows",
        battery_pct=80.0,
        charging_status="Discharging",
        battery_voltage_v=11.4,
        battery_current_a=1.0,
        battery_power_w=11.4,
        discharge_power_w=11.4,
        energy_remaining_wh=33.6,
        energy_full_wh=42.0,
        cpu_util_pct=20.0,
        cpu_freq_mhz=2200.0,
        ram_util_pct=45.0,
        ram_used_gb=7.2,
        gpu_util_pct=10.0,
        power_profile="balanced",
        workload_label="Browsing",
        is_synthetic=True
    )

    pred = predictor.predict_sample(sample)
    assert "predicted_power_w" in pred
    assert pred["predicted_power_w"] > 0.0
    assert pred["is_valid"] is True


def test_battery_runtime_estimator():
    estimator = BatteryRuntimeEstimator()

    discharging_sample = TelemetrySample(
        timestamp=1000.0,
        datetime_str="2026-10-09T00:00:00",
        os_name="Windows",
        battery_pct=50.0,
        charging_status="Discharging",
        battery_voltage_v=11.2,
        battery_current_a=0.9,
        battery_power_w=10.0,
        discharge_power_w=10.0,
        energy_remaining_wh=21.0, # 21 Wh / 10 W = 2.1 hours = 2h 06m
        energy_full_wh=42.0,
        cpu_util_pct=15.0,
        cpu_freq_mhz=2000.0,
        ram_util_pct=40.0,
        ram_used_gb=6.4,
        gpu_util_pct=5.0,
        power_profile="balanced",
        workload_label="Idle",
        is_synthetic=True
    )

    res = estimator.estimate_runtime(discharging_sample, predicted_power_w=10.0)
    assert res["is_charging"] is False
    assert res["runtime_hours"] > 1.5
    assert "2h" in res["runtime_formatted"]

    charging_sample = TelemetrySample(
        timestamp=1000.0,
        datetime_str="2026-10-09T00:00:00",
        os_name="Windows",
        battery_pct=50.0,
        charging_status="Charging",
        battery_voltage_v=12.0,
        battery_current_a=1.5,
        battery_power_w=18.0,
        discharge_power_w=0.0,
        energy_remaining_wh=21.0,
        energy_full_wh=42.0,
        cpu_util_pct=15.0,
        cpu_freq_mhz=2000.0,
        ram_util_pct=40.0,
        ram_used_gb=6.4,
        gpu_util_pct=5.0,
        power_profile="balanced",
        workload_label="Idle",
        is_synthetic=True
    )

    res_charging = estimator.estimate_runtime(charging_sample)
    assert res_charging["is_charging"] is True
    assert "Plugged in" in res_charging["runtime_formatted"]
