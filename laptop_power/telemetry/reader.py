"""
Cross-Platform Telemetry Reader.
Auto-detects environment, gathers hardware metrics via psutil and OS modules,
handles missing values and missing sensors gracefully, and normalizes output.
"""

import os
import sys
import platform
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import psutil
except ImportError:
    psutil = None

from .schema import TelemetrySample
from .windows import WindowsTelemetryCollector
from .linux import LinuxTelemetryCollector
import config

logger = logging.getLogger(__name__)


class TelemetryReader:
    def __init__(self):
        self.os_name = platform.system()
        self._win_collector = WindowsTelemetryCollector() if self.os_name == "Windows" else None
        self._linux_collector = LinuxTelemetryCollector() if self.os_name == "Linux" else None

        # Tracking for energy-rate estimation over time
        self._last_sample_time: Optional[float] = None
        self._last_battery_pct: Optional[float] = None

    def read_sample(self, workload_label: str = "Idle") -> TelemetrySample:
        """Reads current telemetry sample across platform sensors."""
        now = time.time()
        datetime_str = datetime.fromtimestamp(now).isoformat()

        # System Metrics via psutil
        cpu_util = 0.0
        cpu_freq = 2000.0
        ram_util = 0.0
        ram_used_gb = 0.0

        if psutil:
            try:
                cpu_util = float(psutil.cpu_percent(interval=None))
                freq_info = psutil.cpu_freq()
                if freq_info and freq_info.current:
                    cpu_freq = float(freq_info.current)
                mem = psutil.virtual_memory()
                ram_util = float(mem.percent)
                ram_used_gb = float(mem.used / (1024 ** 3))
            except Exception as e:
                logger.debug(f"psutil sensor query error: {e}")

        # GPU metrics (approximate or dummy fallback if no NVML/AMD sysfs API bound)
        gpu_util = self._estimate_gpu_util(cpu_util, workload_label)

        # OS-level Battery Metrics
        charging_status = "Unknown"
        battery_pct = 100.0
        voltage_v = None
        current_a = None
        power_w = None
        discharge_power_w = None
        energy_rem_wh = None
        energy_full_wh = None
        power_profile = "balanced"

        # Check psutil battery first for basic info
        if psutil and hasattr(psutil, "sensors_battery"):
            try:
                bat = psutil.sensors_battery()
                if bat:
                    battery_pct = float(bat.percent)
                    charging_status = "Charging" if bat.power_plugged else "Discharging"
            except Exception:
                pass

        if self.os_name == "Windows" and self._win_collector:
            win_info = self._win_collector.get_power_status_ctypes()
            if win_info.get("charging_status") != "Unknown":
                charging_status = win_info["charging_status"]
            if win_info.get("battery_pct") is not None:
                battery_pct = win_info["battery_pct"]

            wmi_info = self._win_collector.get_wmi_battery_info()
            voltage_v = wmi_info.get("voltage_v") or config.DESIGN_VOLTAGE_V
            power_w = wmi_info.get("power_w")
            discharge_power_w = wmi_info.get("discharge_rate_w")
            power_profile = wmi_info.get("power_profile", "balanced")

        elif self.os_name == "Linux" and self._linux_collector:
            lin_info = self._linux_collector.get_linux_battery_info()
            charging_status = lin_info.get("charging_status", charging_status)
            battery_pct = lin_info.get("battery_pct", battery_pct)
            voltage_v = lin_info.get("voltage_v")
            current_a = lin_info.get("current_a")
            power_w = lin_info.get("power_w")
            discharge_power_w = lin_info.get("discharge_rate_w")
            energy_rem_wh = lin_info.get("energy_remaining_wh")
            energy_full_wh = lin_info.get("energy_full_wh")
            power_profile = lin_info.get("power_profile", "balanced")

        # Handle voltage/current calculations if power not directly reported
        if power_w is None and voltage_v is not None and current_a is not None:
            power_w = voltage_v * current_a

        # If discharging and discharge_power_w is not set but power_w is available
        if charging_status == "Discharging":
            if discharge_power_w is None and power_w is not None and power_w > 0:
                discharge_power_w = power_w
            elif discharge_power_w is None:
                # Estimate discharge rate via percentage delta over time if on battery
                discharge_power_w = self._estimate_power_from_pct(battery_pct, now)
        else:
            discharge_power_w = 0.0

        # Estimated energy remaining if missing
        if energy_rem_wh is None:
            full_cap = energy_full_wh or config.DESIGN_BATTERY_CAPACITY_WH
            energy_rem_wh = (battery_pct / 100.0) * full_cap

        if energy_full_wh is None:
            energy_full_wh = config.DESIGN_BATTERY_CAPACITY_WH

        return TelemetrySample(
            timestamp=now,
            datetime_str=datetime_str,
            os_name=self.os_name,
            battery_pct=round(battery_pct, 2),
            charging_status=charging_status,
            battery_voltage_v=round(voltage_v, 3) if voltage_v else None,
            battery_current_a=round(current_a, 3) if current_a else None,
            battery_power_w=round(power_w, 3) if power_w else None,
            discharge_power_w=round(discharge_power_w, 3) if discharge_power_w is not None else None,
            energy_remaining_wh=round(energy_rem_wh, 3) if energy_rem_wh else None,
            energy_full_wh=round(energy_full_wh, 3) if energy_full_wh else None,
            cpu_util_pct=round(cpu_util, 2),
            cpu_freq_mhz=round(cpu_freq, 1),
            ram_util_pct=round(ram_util, 2),
            ram_used_gb=round(ram_used_gb, 2),
            gpu_util_pct=round(gpu_util, 2),
            power_profile=power_profile,
            workload_label=workload_label,
            is_synthetic=False
        )

    def _estimate_gpu_util(self, cpu_util: float, workload: str) -> float:
        """Estimates GPU utilization based on workload hint when hardware GPU counters are unexposed."""
        if workload in config.WORKLOAD_PRESETS:
            g_min, g_max = config.WORKLOAD_PRESETS[workload]["gpu_range"]
            return round((g_min + g_max) / 2.0, 2)
        return round(max(0.0, cpu_util * 0.2), 2)

    def _estimate_power_from_pct(self, current_pct: float, current_time: float) -> Optional[float]:
        """Calculates approximate power drain from battery percentage decrease over time."""
        estimated_w = None
        if self._last_sample_time and self._last_battery_pct:
            dt_hours = (current_time - self._last_sample_time) / 3600.0
            pct_drop = self._last_battery_pct - current_pct
            if dt_hours > 0.001 and 0.0 < pct_drop < 10.0:
                d_wh = (pct_drop / 100.0) * config.DESIGN_BATTERY_CAPACITY_WH
                estimated_w = d_wh / dt_hours

        self._last_sample_time = current_time
        self._last_battery_pct = current_pct
        return estimated_w
