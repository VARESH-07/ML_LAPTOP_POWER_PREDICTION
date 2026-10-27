"""
Model Storage & Serialization.
Handles saving and loading of fitted ML pipelines with metadata sidecars.
"""

from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline

import config

logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.default_model_path = self.model_dir / "power_model_latest.joblib"
        self.default_meta_path = self.model_dir / "power_model_latest_meta.json"

    def save_model(
        self,
        pipeline: Pipeline,
        metrics: Dict[str, float],
        model_name: str,
        num_samples: int,
        is_synthetic: bool = False,
        path: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Saves model pipeline and metadata JSON.
        """
        model_path = path or self.default_model_path
        meta_path = model_path.with_name(model_path.stem + "_meta.json")

        joblib.dump(pipeline, model_path)

        metadata = {
            "model_name": model_name,
            "trained_at": datetime.now().isoformat(),
            "target_column": config.TARGET_COLUMN,
            "target_unit": "Watts",
            "features": config.FEATURE_COLUMNS,
            "numerical_features": config.NUMERICAL_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "metrics": metrics,
            "num_training_samples": num_samples,
            "is_synthetic_data": is_synthetic,
            "pipeline_class": str(pipeline.__class__.__name__)
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model pipeline to {model_path} and metadata to {meta_path}")
        return model_path, meta_path

    def load_model(self, path: Optional[Path] = None) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
        """
        Loads fitted pipeline and metadata.
        """
        model_path = path or self.default_model_path
        meta_path = model_path.with_name(model_path.stem + "_meta.json")

        if not model_path.exists():
            logger.warning(f"Model file {model_path} does not exist.")
            return None, None

        pipeline = joblib.load(model_path)
        metadata = None
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return pipeline, metadata
