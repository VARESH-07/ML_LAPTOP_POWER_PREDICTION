"""
Power Optimization Recommendation Engine.
Analyzes telemetry metrics, workload state, and power profiles to generate data-driven advisory recommendations.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

from laptop_power.telemetry import TelemetrySample
import config


@dataclass
class Recommendation:
    id: str
    category: str           # 'Power Profile', 'CPU/Workload', 'Display/GPU', 'Session Comparison'
    title: str
    description: str
    potential_savings_w: float
    estimated_battery_extension_min: int
    severity: str           # 'High', 'Medium', 'Low', 'Info'
    actionable: bool
    command_hint: Optional[str] = None


class RecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(
        self,
        sample: TelemetrySample,
        recent_history_df: Optional[pd.DataFrame] = None
    ) -> List[Recommendation]:
        """
        Evaluates current telemetry sample and optional historical dataframe to generate advisory power saving tips.
        """
        recs: List[Recommendation] = []

        # 1. Power Profile Recommendation
        if sample.charging_status == "Discharging":
            if sample.power_profile in ["performance", "balanced", "high performance"]:
                power_diff_w = 3.5 if sample.power_profile == "performance" else 1.8
                battery_ext_min = self._calc_extension_minutes(sample, power_diff_w)

                cmd = "powerprofilesctl set power-saver" if sample.os_name == "Linux" else "powercfg /setactive SCHEME_MAXPOWER"

                recs.append(Recommendation(
                    id="rec_power_profile",
                    category="Power Profile",
                    title="Switch to Power Saver Profile",
                    description=f"Active profile is currently '{sample.power_profile}'. Switching to 'power-saver' lowers CPU boost clocks and background power draw.",
                    potential_savings_w=power_diff_w,
                    estimated_battery_extension_min=battery_ext_min,
                    severity="High" if sample.battery_pct < 30.0 else "Medium",
                    actionable=True,
                    command_hint=cmd
                ))

        # 2. CPU / Workload Throttling Recommendation
        if sample.cpu_util_pct > 60.0:
            power_diff_w = (sample.cpu_util_pct / 100.0) * 8.0
            battery_ext_min = self._calc_extension_minutes(sample, power_diff_w)

            recs.append(Recommendation(
                id="rec_cpu_high",
                category="CPU/Workload",
                title="Sustained High CPU Usage Detected",
                description=f"CPU utilization is at {sample.cpu_util_pct:.1f}%. High CPU frequency ({sample.cpu_freq_mhz:.0f} MHz) increases thermal and power output.",
                potential_savings_w=round(power_diff_w, 1),
                estimated_battery_extension_min=battery_ext_min,
                severity="High" if sample.cpu_util_pct > 80.0 else "Medium",
                actionable=False,
                command_hint="Inspect process manager for unnecessary background worker threads."
            ))

        # 3. GPU / Graphics Acceleration Recommendation
        if sample.gpu_util_pct > 40.0:
            power_diff_w = (sample.gpu_util_pct / 100.0) * 5.0
            battery_ext_min = self._calc_extension_minutes(sample, power_diff_w)

            recs.append(Recommendation(
                id="rec_gpu_high",
                category="Display/GPU",
                title="Active GPU Workload Detected",
                description=f"GPU utilization is {sample.gpu_util_pct:.1f}%. Consider disabling hardware acceleration or closing 3D/video apps on battery.",
                potential_savings_w=round(power_diff_w, 1),
                estimated_battery_extension_min=battery_ext_min,
                severity="Medium",
                actionable=False
            ))

        # 4. Display Brightness Hint
        if sample.charging_status == "Discharging" and (sample.discharge_power_w or 10.0) > 12.0:
            recs.append(Recommendation(
                id="rec_brightness",
                category="Display/GPU",
                title="Reduce Screen Brightness",
                description="Display panel backlight is a major power consumer. Reducing screen brightness by 30% can save ~1.5W to 2.5W.",
                potential_savings_w=2.0,
                estimated_battery_extension_min=self._calc_extension_minutes(sample, 2.0),
                severity="Low",
                actionable=True
            ))

        # 5. Historical Workload Session Comparison
        if recent_history_df is not None and not recent_history_df.empty and "workload_label" in recent_history_df.columns:
            workload_means = recent_history_df.groupby("workload_label")["discharge_power_w"].mean().dropna().to_dict()
            if len(workload_means) > 1:
                min_workload = min(workload_means, key=workload_means.get)
                max_workload = max(workload_means, key=workload_means.get)
                diff = workload_means[max_workload] - workload_means[min_workload]

                if diff > 3.0:
                    ext = self._calc_extension_minutes(sample, diff)
                    recs.append(Recommendation(
                        id="rec_session_comp",
                        category="Session Comparison",
                        title=f"Workload Drain Contrast ({max_workload} vs {min_workload})",
                        description=f"Measured session history shows '{max_workload}' draws ~{workload_means[max_workload]:.1f}W vs '{min_workload}' at ~{workload_means[min_workload]:.1f}W.",
                        potential_savings_w=round(diff, 1),
                        estimated_battery_extension_min=ext,
                        severity="Info",
                        actionable=False
                    ))

        return recs

    def _calc_extension_minutes(self, sample: TelemetrySample, saved_power_w: float) -> int:
        """Estimates added battery runtime in minutes if power draw is reduced by saved_power_w."""
        if sample.charging_status != "Discharging":
            return 0

        curr_power = sample.discharge_power_w or 10.0
        if curr_power <= saved_power_w or curr_power <= 1.0:
            return 0

        energy_wh = sample.energy_remaining_wh or ((sample.battery_pct / 100.0) * config.DESIGN_BATTERY_CAPACITY_WH)
        curr_runtime_h = energy_wh / curr_power
        new_runtime_h = energy_wh / (curr_power - saved_power_w)
        ext_h = max(0.0, new_runtime_h - curr_runtime_h)

        return int(ext_h * 60)
