"""
Dataset Manager.
Loads, validates, cleans, and merges recorded session CSV datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional

import config
from .synthetic import generate_synthetic_dataset

logger = logging.getLogger(__name__)


class DatasetManager:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.RAW_DATA_DIR

    def load_combined_dataset(self, include_synthetic_fallback: bool = True) -> pd.DataFrame:
        """
        Loads all CSV files in raw data directory.
        If no files exist and include_synthetic_fallback is True, generates a synthetic dataset.
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            if include_synthetic_fallback:
                logger.info("No raw session CSV datasets found. Generating synthetic dataset baseline...")
                syn_path = self.data_dir / "synthetic_default.csv"
                return generate_synthetic_dataset(num_samples=1200, output_path=syn_path)
            else:
                return pd.DataFrame()

        dfs = []
        for f in csv_files:
            try:
                df_f = pd.read_csv(f)
                if not df_f.empty and "discharge_power_w" in df_f.columns:
                    dfs.append(df_f)
            except Exception as e:
                logger.warning(f"Could not read dataset file {f}: {e}")

        if not dfs:
            if include_synthetic_fallback:
                syn_path = self.data_dir / "synthetic_default.csv"
                return generate_synthetic_dataset(num_samples=1200, output_path=syn_path)
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        return self.clean_and_validate(combined)

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates telemetry schema, drops exact duplicate timestamps, filters physically impossible outliers.
        """
        if df.empty:
            return df

        # Deduplicate by timestamp if present
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"]).copy()

        # Sort chronologically
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp").reset_index(drop=True)

        # Ensure required numerical columns exist
        for num_col in config.NUMERICAL_FEATURES:
            if num_col not in df.columns:
                df[num_col] = 0.0
            else:
                df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)

        for cat_col in config.CATEGORICAL_FEATURES:
            if cat_col not in df.columns:
                df[cat_col] = "unknown" if cat_col != "workload_label" else "Idle"
            else:
                df[cat_col] = df[cat_col].fillna("unknown").astype(str)

        # Filter target outliers for supervised learning (discharging power between 0.5W and 120W)
        if config.TARGET_COLUMN in df.columns:
            # Keep rows with valid target or when discharging
            valid_target_mask = (df[config.TARGET_COLUMN] >= 0.0) & (df[config.TARGET_COLUMN] <= 120.0)
            df = df[valid_target_mask].copy()

        return df

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset session-by-session or chronologically to prevent temporal data leakage.
        """
        if df.empty:
            return df, df

        # Filter out charging samples where discharge power is 0.0 if predicting discharge power
        discharging_df = df[df["charging_status"] == "Discharging"].copy()
        if len(discharging_df) < 50:
            # Fallback to full df if small dataset
            discharging_df = df.copy()

        if "session_id" in discharging_df.columns and len(discharging_df["session_id"].unique()) > 1:
            sessions = discharging_df["session_id"].unique()
            num_test = max(1, int(len(sessions) * test_ratio))
            test_sessions = set(sessions[-num_test:])
            train_df = discharging_df[~discharging_df["session_id"].isin(test_sessions)]
            test_df = discharging_df[discharging_df["session_id"].isin(test_sessions)]
        else:
            # Chronological split
            split_idx = int(len(discharging_df) * (1 - test_ratio))
            train_df = discharging_df.iloc[:split_idx]
            test_df = discharging_df.iloc[split_idx:]

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
