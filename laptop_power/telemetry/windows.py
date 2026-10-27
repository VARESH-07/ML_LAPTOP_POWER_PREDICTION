"""
Windows Telemetry Collector.
Uses psutil, ctypes GetSystemPowerStatus, and PowerShell/WMI fallback to read battery and hardware metrics.
"""

import ctypes
from ctypes import wintypes
import logging
import subprocess
import json
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ('ACLineStatus', wintypes.BYTE),
        ('BatteryFlag', wintypes.BYTE),
        ('BatteryLifePercent', wintypes.BYTE),
        ('SystemStatusFlag', wintypes.BYTE),
        ('BatteryLifeTime', wintypes.DWORD),
        ('BatteryFullLifeTime', wintypes.DWORD),
    ]


class WindowsTelemetryCollector:
    def __init__(self):
        self._last_energy_wh: Optional[float] = None
        self._last_energy_time: Optional[float] = None

    def get_power_status_ctypes(self) -> Dict[str, Any]:
        status = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            ac_status = status.ACLineStatus
            if ac_status == 1:
                charging_status = "Charging"
            elif ac_status == 0:
                charging_status = "Discharging"
            else:
                charging_status = "Unknown"

            pct = status.BatteryLifePercent if status.BatteryLifePercent <= 100 else 100.0
            runtime_sec = status.BatteryLifeTime if status.BatteryLifeTime != 0xFFFFFFFF else None
            return {
                "charging_status": charging_status,
                "battery_pct": float(pct),
                "runtime_sec": runtime_sec,
            }
        return {"charging_status": "Unknown", "battery_pct": 50.0, "runtime_sec": None}

    def get_wmi_battery_info(self) -> Dict[str, Any]:
        """Queries WMI/CIM via PowerShell for voltage, discharge rate, and capacity."""
        info = {
            "voltage_v": None,
            "discharge_rate_w": None,
            "charge_rate_w": None,
            "power_w": None,
            "power_profile": "balanced"
        }
        try:
            ps_cmd = (
                "Get-CimInstance -ClassName Win32_Battery | "
                "Select-Object DesignVoltage, EstimatedChargeRemaining, BatteryStatus | ConvertTo-Json"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                if isinstance(data, dict):
                    voltage_mv = data.get("DesignVoltage")
                    if voltage_mv and isinstance(voltage_mv, (int, float)) and voltage_mv > 0:
                        info["voltage_v"] = float(voltage_mv) / 1000.0
        except Exception as e:
            logger.debug(f"WMI Win32_Battery query failed: {e}")

        # Try root/wmi BatteryStatus for discharge rate in mW
        try:
            ps_cmd_wmi = (
                "Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
                "Select-Object DischargeRate, ChargeRate, Voltage | ConvertTo-Json"
            )
            res_wmi = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd_wmi],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res_wmi.returncode == 0 and res_wmi.stdout.strip():
                wmi_data = json.loads(res_wmi.stdout)
                if isinstance(wmi_data, list) and len(wmi_data) > 0:
                    wmi_data = wmi_data[0]
                if isinstance(wmi_data, dict):
                    discharge_mw = wmi_data.get("DischargeRate")
                    charge_mw = wmi_data.get("ChargeRate")
                    v_mv = wmi_data.get("Voltage")

                    if v_mv and v_mv > 0:
                        info["voltage_v"] = float(v_mv) / 1000.0

                    if discharge_mw and discharge_mw > 0:
                        info["discharge_rate_w"] = float(discharge_mw) / 1000.0
                        info["power_w"] = info["discharge_rate_w"]
                    elif charge_mw and charge_mw > 0:
                        info["charge_rate_w"] = float(charge_mw) / 1000.0
                        info["power_w"] = info["charge_rate_w"]
        except Exception as e:
            logger.debug(f"WMI root/wmi BatteryStatus query failed: {e}")

        # Get active power plan via powercfg
        try:
            res_pcfg = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if res_pcfg.returncode == 0:
                out = res_pcfg.stdout.lower()
                if "power saver" in out or "saver" in out:
                    info["power_profile"] = "power-saver"
                elif "high performance" in out or "performance" in out:
                    info["power_profile"] = "performance"
                else:
                    info["power_profile"] = "balanced"
        except Exception as e:
            logger.debug(f"powercfg query failed: {e}")

        return info
