"""
ML Models, Evaluation, and Serialization Unit Tests.
"""
import pytest
import numpy as np
from pathlib import Path
from laptop_power.data import generate_synthetic_dataset, DatasetManager
from laptop_power.models import ModelTrainer, ModelEvaluator, ModelStorage


def test_model_evaluator():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0])

    metrics = ModelEvaluator.evaluate(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] == 1.25
    assert metrics["r2"] > 0.9

    res_df = ModelEvaluator.calculate_residuals(y_true, y_pred)
    assert len(res_df) == 4
    assert "residual_w" in res_df.columns


def test_model_trainer_and_storage(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()

    syn_df = generate_synthetic_dataset(num_samples=300, output_path=data_dir / "syn.csv")
    dataset_mgr = DatasetManager(data_dir=data_dir)
    storage = ModelStorage(model_dir=model_dir)

    trainer = ModelTrainer(data_manager=dataset_mgr, storage=storage)
    best_pipe, best_metrics, comp_df = trainer.train_and_evaluate_all(syn_df)

    assert best_pipe is not None
    assert "mae" in best_metrics
    assert len(comp_df) == 4

    # Test storage reload
    reloaded_pipe, reloaded_meta = storage.load_model()
    assert reloaded_pipe is not None
    assert reloaded_meta["model_name"] == comp_df.iloc[0]["Model"]
