"""
Data and Synthetic Pipeline Unit Tests.
"""
import pytest
from pathlib import Path
import pandas as pd
from laptop_power.data import generate_synthetic_dataset, DatasetManager, DataCollector
import config


def test_generate_synthetic_dataset(tmp_path):
    out_file = tmp_path / "syn_test.csv"
    df = generate_synthetic_dataset(num_samples=100, output_path=out_file)
    assert len(df) == 100
    assert out_file.exists()
    assert "discharge_power_w" in df.columns
    assert "cpu_util_pct" in df.columns
    assert df["is_synthetic"].all()


def test_dataset_manager_clean_and_split(tmp_path):
    df_syn = generate_synthetic_dataset(num_samples=200, output_path=tmp_path / "test.csv")
    mgr = DatasetManager(data_dir=tmp_path)

    cleaned = mgr.clean_and_validate(df_syn)
    assert not cleaned.empty

    train_df, test_df = mgr.prepare_train_test_split(cleaned, test_ratio=0.2)
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(cleaned[cleaned["charging_status"] == "Discharging"])


def test_data_collector_single_sample():
    collector = DataCollector()
    sample = collector.collect_single_sample(workload_label="Browsing")
    assert sample.workload_label == "Browsing"
    assert isinstance(sample.battery_pct, float)
