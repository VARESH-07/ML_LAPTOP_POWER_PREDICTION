"""
Model Trainer.
Trains and compares candidate ML regression models (Dummy, Ridge, RandomForest, HistGradientBoosting)
on laptop power telemetry datasets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from laptop_power.data import DatasetManager
from laptop_power.features import build_feature_pipeline, prepare_features_and_target
from .evaluator import ModelEvaluator
from .storage import ModelStorage
import config

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_manager: Optional[DatasetManager] = None, storage: Optional[ModelStorage] = None):
        self.data_manager = data_manager or DatasetManager()
        self.storage = storage or ModelStorage()

    def train_and_evaluate_all(self, df: Optional[pd.DataFrame] = None) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        """
        Trains candidate models, selects the best performing pipeline on held-out test data,
        saves the pipeline, and returns (best_pipeline, evaluation_metrics, candidate_comparison_df).
        """
        if df is None or df.empty:
            df = self.data_manager.load_combined_dataset(include_synthetic_fallback=True)

        is_synthetic = bool(df["is_synthetic"].iloc[0]) if "is_synthetic" in df.columns and not df.empty else True

        # Split session/chronological train and test
        train_df, test_df = self.data_manager.prepare_train_test_split(df, test_ratio=0.2)

        X_train, y_train = prepare_features_and_target(train_df)
        X_test, y_test = prepare_features_and_target(test_df)

        if len(X_train) == 0:
            raise ValueError("Training dataset is empty after preprocessing.")

        candidates = {
            "DummyRegressor Baseline": DummyRegressor(strategy="mean"),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=2),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=42)
        }

        results = []
        best_model_name = None
        best_pipeline = None
        best_mae = float("inf")
        best_metrics = {}

        for name, model in candidates.items():
            preprocessor = build_feature_pipeline()
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            # Fit on train
            pipeline.fit(X_train, y_train)

            # Predict on test
            y_pred = pipeline.predict(X_test)
            metrics = ModelEvaluator.evaluate(y_test.values, y_pred)

            results.append({
                "Model": name,
                "MAE (W)": metrics["mae"],
                "RMSE (W)": metrics["rmse"],
                "R² Score": metrics["r2"],
                "Max Error (W)": metrics["max_error"]
            })

            # Select model with lowest MAE
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_model_name = name
                best_pipeline = pipeline
                best_metrics = metrics

        comparison_df = pd.DataFrame(results).sort_values(by="MAE (W)").reset_index(drop=True)
        logger.info(f"Selected best model: {best_model_name} with MAE: {best_mae:.4f} W")

        # Save selected best pipeline
        self.storage.save_model(
            pipeline=best_pipeline,
            metrics=best_metrics,
            model_name=best_model_name,
            num_samples=len(df),
            is_synthetic=is_synthetic
        )

        return best_pipeline, best_metrics, comparison_df
