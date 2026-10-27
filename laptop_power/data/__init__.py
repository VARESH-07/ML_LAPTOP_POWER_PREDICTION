"""
Data management and synthetic dataset generation subpackage.
"""
from .collector import DataCollector
from .dataset import DatasetManager
from .synthetic import generate_synthetic_dataset

__all__ = ["DataCollector", "DatasetManager", "generate_synthetic_dataset"]
