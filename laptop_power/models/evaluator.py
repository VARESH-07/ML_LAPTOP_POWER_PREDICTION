"""
Model Evaluator.
Calculates regression evaluation metrics (MAE, RMSE, R2, MAPE) and residual stats.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes regression metrics MAE, RMSE, R2, and Max Error.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "max_error": 0.0}

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
        max_err = float(np.max(np.abs(y_true - y_pred)))

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "max_error": round(max_err, 4),
        }

    @staticmethod
    def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Returns a DataFrame containing actual, predicted, residual, and percentage error.
        """
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        pct_error = np.where(y_true > 0, (abs_residuals / y_true) * 100.0, 0.0)

        return pd.DataFrame({
            "actual_w": y_true,
            "predicted_w": y_pred,
            "residual_w": residuals,
            "abs_error_w": abs_residuals,
            "pct_error": np.round(pct_error, 2)
        })
