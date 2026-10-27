"""
Telemetry Schema and Data Structures.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time


@dataclass
class TelemetrySample:
    timestamp: float                    # Unix timestamp (seconds)
    datetime_str: str                   # ISO formatted datetime string
    os_name: str                        # 'Windows', 'Linux', 'Darwin', etc.
    battery_pct: float                  # Battery percentage 0-100
    charging_status: str                # 'Discharging', 'Charging', 'Full', 'Unknown'
    battery_voltage_v: Optional[float]  # Voltage in Volts
    battery_current_a: Optional[float]  # Current in Amperes (positive for discharge, negative for charge or abs)
    battery_power_w: Optional[float]    # Instantaneous battery power in Watts (observed directly or V * A)
    discharge_power_w: Optional[float]  # Positive discharge rate in Watts (0.0 if plugged in / charging)
    energy_remaining_wh: Optional[float]# Energy remaining in Wh
    energy_full_wh: Optional[float]     # Energy capacity full in Wh
    cpu_util_pct: float                 # System CPU utilization %
    cpu_freq_mhz: float                 # Primary CPU frequency in MHz
    ram_util_pct: float                 # System RAM utilization %
    ram_used_gb: float                  # RAM used in GB
    gpu_util_pct: float                 # Integrated/Discrete GPU utilization %
    power_profile: str                  # Power profile ('balanced', 'power-saver', 'performance', etc.)
    workload_label: str                 # User or estimated workload label
    is_synthetic: bool = False          # Flag indicating if this sample is synthetic/simulated

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetrySample":
        return cls(**data)
