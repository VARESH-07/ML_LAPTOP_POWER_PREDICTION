"""
CLI Script to Train, Evaluate, and Save Machine Learning Models.
"""

import sys
import logging
from pathlib import Path

# Ensure package import
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from laptop_power.data import DatasetManager
from laptop_power.models import ModelTrainer, ModelStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Dataset Manager and Model Storage...")
    dataset_mgr = DatasetManager()
    storage = ModelStorage()

    combined_df = dataset_mgr.load_combined_dataset(include_synthetic_fallback=True)
    logger.info(f"Loaded dataset with {len(combined_df)} samples.")

    trainer = ModelTrainer(data_manager=dataset_mgr, storage=storage)
    logger.info("Starting candidate model training and comparison...")

    best_pipe, best_metrics, comparison_df = trainer.train_and_evaluate_all(combined_df)

    print("\n" + "=" * 60)
    print(" CANDIDATE MODEL EVALUATION COMPARISON ")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    print(f"BEST MODEL DEPLOYED: {comparison_df.iloc[0]['Model']}")
    print(f"HELD-OUT MAE: {best_metrics['mae']:.4f} Watts")
    print(f"HELD-OUT RMSE: {best_metrics['rmse']:.4f} Watts")
    print(f"HELD-OUT R² SCORE: {best_metrics['r2']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
