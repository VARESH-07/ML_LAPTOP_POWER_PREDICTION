"""
Battery Runtime Estimator.
Calculates smoothed remaining battery runtime using energy capacity and discharge power estimates.
Implements exponential moving average (EMA) smoothing and uncertainty warnings.
"""

from collections import deque
import logging
from typing import Dict, Any, Optional

from laptop_power.telemetry import TelemetrySample
import config

logger = logging.getLogger(__name__)


class BatteryRuntimeEstimator:
    def __init__(self, window_size: int = 6, alpha: float = 0.3):
        self.window_size = window_size
        self.alpha = alpha  # EMA smoothing factor
        self.power_history = deque(maxlen=window_size)
        self.smoothed_power_w: Optional[float] = None

    def reset(self):
        """Resets power history deque and smoothed power state."""
        self.power_history.clear()
        self.smoothed_power_w = None

    def estimate_runtime(self, sample: TelemetrySample, predicted_power_w: Optional[float] = None) -> Dict[str, Any]:
        """
        Estimates remaining battery runtime based on live telemetry and optional ML predicted power.
        """
        if sample.charging_status == "Charging":
            return {
                "runtime_hours": None,
                "runtime_minutes": None,
                "runtime_formatted": "Plugged in (Charging)",
                "smoothed_power_w": 0.0,
                "confidence": "High",
                "is_charging": True,
                "warning": None
            }

        if sample.charging_status == "Full":
            return {
                "runtime_hours": None,
                "runtime_minutes": None,
                "runtime_formatted": "Plugged in (Full)",
                "smoothed_power_w": 0.0,
                "confidence": "High",
                "is_charging": True,
                "warning": None
            }

        # Select discharge power: prefer directly observed, then predicted, then default baseline
        raw_power = sample.discharge_power_w
        if raw_power is None or raw_power <= 0.5:
            if predicted_power_w and predicted_power_w > 0.5:
                raw_power = predicted_power_w
            else:
                raw_power = 10.0  # Fallback 10W baseline draw

        # Update EMA smoothed discharge power
        if self.smoothed_power_w is None:
            self.smoothed_power_w = raw_power
        else:
            self.smoothed_power_w = (self.alpha * raw_power) + ((1 - self.alpha) * self.smoothed_power_w)

        self.power_history.append(raw_power)

        # Usable energy in Wh
        energy_wh = sample.energy_remaining_wh
        uncertainty_warning = None

        if energy_wh is None or energy_wh <= 0:
            # Fallback based on battery %
            full_wh = sample.energy_full_wh or config.DESIGN_BATTERY_CAPACITY_WH
            energy_wh = (sample.battery_pct / 100.0) * full_wh
            uncertainty_warning = "Remaining energy (Wh) unexposed by hardware; estimated from battery percentage."

        if self.smoothed_power_w <= 0.1:
            return {
                "runtime_hours": None,
                "runtime_minutes": None,
                "runtime_formatted": "Calculating...",
                "smoothed_power_w": 0.0,
                "confidence": "Low",
                "is_charging": False,
                "warning": "Discharge power reading is near zero."
            }

        runtime_hours = energy_wh / self.smoothed_power_w
        # Cap reasonable runtime between 0 and 24 hours
        runtime_hours = min(24.0, max(0.0, runtime_hours))
        total_minutes = int(runtime_hours * 60)
        hrs = total_minutes // 60
        mins = total_minutes % 60

        # Confidence rating based on sample history stability
        if len(self.power_history) >= 4:
            std_dev = float(np_std(list(self.power_history))) if len(self.power_history) > 1 else 0.0
            confidence = "High" if std_dev < 2.0 else ("Medium" if std_dev < 5.0 else "Low")
        else:
            confidence = "Medium"

        return {
            "runtime_hours": round(runtime_hours, 2),
            "runtime_minutes": total_minutes,
            "runtime_formatted": f"{hrs}h {mins:02d}m",
            "smoothed_power_w": round(self.smoothed_power_w, 2),
            "confidence": confidence,
            "is_charging": False,
            "warning": uncertainty_warning
        }


def np_std(vals):
    import numpy as np
    return np.std(vals)
