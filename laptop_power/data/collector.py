"""
Telemetry Data Collector.
Manages telemetry logging sessions, dataset persistence, schema validation, and outlier handling.
"""

import threading
import time
import logging
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any, Callable

from laptop_power.telemetry import TelemetryReader, TelemetrySample
import config

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, output_dir: Optional[Path] = None, interval_sec: int = config.DEFAULT_SAMPLE_INTERVAL_SEC):
        self.output_dir = output_dir or config.RAW_DATA_DIR
        self.interval_sec = interval_sec
        self.reader = TelemetryReader()

        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self.current_session_id: Optional[str] = None
        self.current_csv_file: Optional[Path] = None
        self.active_workload_label: str = "Idle"
        self._samples_collected_in_session = 0
        self._on_sample_callback: Optional[Callable[[TelemetrySample], None]] = None

    def start_session(self, workload_label: str = "Idle", callback: Optional[Callable[[TelemetrySample], None]] = None) -> str:
        """Starts a new telemetry collection session."""
        if self._is_running:
            logger.warning("Data collection session already running.")
            return self.current_session_id or "active_session"

        self.active_workload_label = workload_label
        self._on_sample_callback = callback
        session_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"session_{session_time_str}"
        self.current_csv_file = self.output_dir / f"{self.current_session_id}.csv"

        self._is_running = True
        self._samples_collected_in_session = 0
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

        logger.info(f"Started collection session {self.current_session_id} -> {self.current_csv_file}")
        return self.current_session_id

    def stop_session(self):
        """Stops the current telemetry collection session safely."""
        if not self._is_running:
            return
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info(f"Stopped collection session {self.current_session_id}. Total samples: {self._samples_collected_in_session}")

    def is_running(self) -> bool:
        return self._is_running

    def collect_single_sample(self, workload_label: Optional[str] = None) -> TelemetrySample:
        """Collects and returns a single telemetry sample without starting a thread."""
        label = workload_label or self.active_workload_label
        return self.reader.read_sample(workload_label=label)

    def _collection_loop(self):
        while self._is_running:
            try:
                sample = self.reader.read_sample(workload_label=self.active_workload_label)
                self._append_sample_to_csv(sample)
                self._samples_collected_in_session += 1

                if self._on_sample_callback:
                    try:
                        self._on_sample_callback(sample)
                    except Exception as cb_err:
                        logger.error(f"Error in sample callback: {cb_err}")

            except Exception as e:
                logger.error(f"Error in telemetry collection loop: {e}")

            time.sleep(self.interval_sec)

    def _append_sample_to_csv(self, sample: TelemetrySample):
        if not self.current_csv_file:
            return

        sample_dict = sample.to_dict()
        sample_dict["session_id"] = self.current_session_id
        df = pd.DataFrame([sample_dict])

        file_exists = self.current_csv_file.exists()
        df.to_csv(self.current_csv_file, mode='a', header=not file_exists, index=False)
