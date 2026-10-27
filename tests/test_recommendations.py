"""
Power Recommendation Engine Unit Tests.
"""
import pytest
from laptop_power.telemetry import TelemetrySample
from laptop_power.recommendations import RecommendationEngine


def test_recommendation_engine():
    engine = RecommendationEngine()

    high_cpu_sample = TelemetrySample(
        timestamp=1000.0,
        datetime_str="2026-10-09T00:00:00",
        os_name="Windows",
        battery_pct=25.0,
        charging_status="Discharging",
        battery_voltage_v=10.8,
        battery_current_a=2.0,
        battery_power_w=21.6,
        discharge_power_w=21.6,
        energy_remaining_wh=10.5,
        energy_full_wh=42.0,
        cpu_util_pct=85.0, # High CPU load
        cpu_freq_mhz=3800.0,
        ram_util_pct=70.0,
        ram_used_gb=11.2,
        gpu_util_pct=10.0,
        power_profile="performance", # Performance profile on battery
        workload_label="Compiling",
        is_synthetic=True
    )

    recs = engine.generate_recommendations(high_cpu_sample)
    assert len(recs) >= 2

    categories = [r.category for r in recs]
    assert "Power Profile" in categories
    assert "CPU/Workload" in categories

    for r in recs:
        assert r.potential_savings_w > 0.0
        assert r.severity in ["High", "Medium", "Low", "Info"]
