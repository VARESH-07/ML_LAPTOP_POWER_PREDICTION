"""
Automated Test Suite for Laptop Power ML Application.
"""
import pytest
from laptop_power.telemetry import TelemetrySample, TelemetryReader


def test_telemetry_sample_dataclass():
    sample = TelemetrySample(
        timestamp=1000.0,
        datetime_str="2026-10-09T00:00:00",
        os_name="Windows",
        battery_pct=85.0,
        charging_status="Discharging",
        battery_voltage_v=11.4,
        battery_current_a=1.0,
        battery_power_w=11.4,
        discharge_power_w=11.4,
        energy_remaining_wh=35.7,
        energy_full_wh=42.0,
        cpu_util_pct=15.0,
        cpu_freq_mhz=2000.0,
        ram_util_pct=50.0,
        ram_used_gb=8.0,
        gpu_util_pct=5.0,
        power_profile="balanced",
        workload_label="Idle",
        is_synthetic=False
    )
    d = sample.to_dict()
    assert d["battery_pct"] == 85.0
    assert d["discharge_power_w"] == 11.4

    reconstructed = TelemetrySample.from_dict(d)
    assert reconstructed.os_name == "Windows"
    assert reconstructed.charging_status == "Discharging"


def test_telemetry_reader_live():
    reader = TelemetryReader()
    sample = reader.read_sample(workload_label="Coding")
    assert isinstance(sample, TelemetrySample)
    assert 0.0 <= sample.battery_pct <= 100.0
    assert sample.cpu_util_pct >= 0.0
    assert sample.ram_util_pct >= 0.0
