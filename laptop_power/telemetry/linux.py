"""
Linux Telemetry Collector.
Reads sysfs paths (/sys/class/power_supply/BAT*), powerprofilesctl, upower, and CPU frequency files.
"""

import os
import glob
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LinuxTelemetryCollector:
    def __init__(self):
        self._bat_path: Optional[str] = self._find_battery_path()

    def _find_battery_path(self) -> Optional[str]:
        paths = glob.glob("/sys/class/power_supply/BAT*")
        if paths:
            return paths[0]
        return None

    def _read_sysfs_val(self, filename: str) -> Optional[float]:
        if not self._bat_path:
            return None
        filepath = os.path.join(self._bat_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    val = f.read().strip()
                    return float(val)
            except Exception as e:
                logger.debug(f"Failed to read sysfs {filepath}: {e}")
        return None

    def _read_sysfs_str(self, filename: str) -> Optional[str]:
        if not self._bat_path:
            return None
        filepath = os.path.join(self._bat_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return f.read().strip()
            except Exception as e:
                logger.debug(f"Failed to read sysfs {filepath}: {e}")
        return None

    def get_linux_battery_info(self) -> Dict[str, Any]:
        info = {
            "charging_status": "Unknown",
            "battery_pct": 50.0,
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "discharge_rate_w": None,
            "energy_remaining_wh": None,
            "energy_full_wh": None,
            "power_profile": "balanced"
        }

        if not self._bat_path:
            return info

        status_str = self._read_sysfs_str("status")
        if status_str:
            info["charging_status"] = status_str

        capacity = self._read_sysfs_val("capacity")
        if capacity is not None:
            info["battery_pct"] = capacity

        # Microvolts to Volts
        voltage_uv = self._read_sysfs_val("voltage_now")
        if voltage_uv is not None:
            info["voltage_v"] = voltage_uv / 1e6

        # Microamps to Amps
        current_ua = self._read_sysfs_val("current_now")
        if current_ua is not None:
            info["current_a"] = abs(current_ua) / 1e6

        # Microwatts to Watts (if directly exposed)
        power_uw = self._read_sysfs_val("power_now")
        if power_uw is not None:
            info["power_w"] = power_uw / 1e6
        elif info["voltage_v"] and info["current_a"]:
            info["power_w"] = info["voltage_v"] * info["current_a"]

        if info["charging_status"] == "Discharging" and info["power_w"]:
            info["discharge_rate_w"] = info["power_w"]

        # Energy remaining / full (Micro-watt hours to Watt hours)
        energy_uwh = self._read_sysfs_val("energy_now")
        if energy_uwh is not None:
            info["energy_remaining_wh"] = energy_uwh / 1e6

        energy_full_uwh = self._read_sysfs_val("energy_full")
        if energy_full_uwh is not None:
            info["energy_full_wh"] = energy_full_uwh / 1e6

        # Check powerprofilesctl
        try:
            res = subprocess.run(
                ["powerprofilesctl", "get"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                info["power_profile"] = res.stdout.strip()
        except Exception:
            pass

        return info
