"""
Feature Pipeline Construction & Preprocessing.
Constructs scikit-learn preprocessing pipelines, handles numerical scaling, categorical encoding,
and feature matrix extraction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config


def build_feature_pipeline() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for telemetry features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                config.NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts feature dataframe X and target series y from dataset, ensuring schema consistency.
    """
    # Ensure all required features are present
    df_clean = df.copy()

    for col in config.NUMERICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0.0

    for col in config.CATEGORICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = "unknown"

    X = df_clean[config.FEATURE_COLUMNS].copy()

    if target_col in df_clean.columns:
        y = df_clean[target_col].astype(float)
    else:
        y = pd.Series(np.zeros(len(df_clean)))

    return X, y
