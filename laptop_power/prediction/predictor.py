"""
Power Consumption Predictor.
Inference engine that transforms input telemetry samples and predicts total power consumption in Watts.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

from laptop_power.telemetry import TelemetrySample
from laptop_power.models import ModelStorage, ModelTrainer
from laptop_power.features import prepare_features_and_target
import config

logger = logging.getLogger(__name__)


class PowerPredictor:
    def __init__(self, storage: Optional[ModelStorage] = None):
        self.storage = storage or ModelStorage()
        self.pipeline = None
        self.metadata = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        pipeline, metadata = self.storage.load_model()
        if pipeline is not None and metadata is not None:
            self.pipeline = pipeline
            self.metadata = metadata
            logger.info(f"Loaded existing model: {metadata.get('model_name')}")
        else:
            logger.info("No trained model found. Auto-training baseline model pipeline...")
            try:
                trainer = ModelTrainer(storage=self.storage)
                self.pipeline, metrics, comparison = trainer.train_and_evaluate_all()
                _, self.metadata = self.storage.load_model()
            except Exception as e:
                logger.error(f"Failed to auto-train baseline model: {e}")

    def predict_sample(self, sample: TelemetrySample) -> Dict[str, Any]:
        """
        Predicts power consumption in Watts for a single telemetry sample.
        """
        sample_dict = sample.to_dict()
        df = pd.DataFrame([sample_dict])
        X, _ = prepare_features_and_target(df)

        if self.pipeline is None:
            # Fallback physics estimate if model unavailable
            est_power = sample.discharge_power_w or (sample.cpu_util_pct * 0.2 + 5.0)
            return {
                "predicted_power_w": round(est_power, 2),
                "is_model_prediction": False,
                "model_name": "Physics Fallback Rule",
                "model_mae": None,
                "unit": "Watts",
                "is_valid": True
            }

        try:
            pred_arr = self.pipeline.predict(X)
            predicted_w = float(np.clip(pred_arr[0], 2.0, 100.0))

            return {
                "predicted_power_w": round(predicted_w, 2),
                "is_model_prediction": True,
                "model_name": self.metadata.get("model_name", "ML Pipeline") if self.metadata else "ML Pipeline",
                "model_mae": self.metadata.get("metrics", {}).get("mae") if self.metadata else None,
                "unit": "Watts",
                "is_valid": True
            }
        except Exception as e:
            logger.error(f"Prediction failed on sample: {e}")
            est_power = sample.discharge_power_w or (sample.cpu_util_pct * 0.2 + 5.0)
            return {
                "predicted_power_w": round(est_power, 2),
                "is_model_prediction": False,
                "model_name": "Error Fallback",
                "model_mae": None,
                "unit": "Watts",
                "is_valid": False
            }
