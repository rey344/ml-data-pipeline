"""Model service.

Handles model training, evaluation, and prediction operations.
"""
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
import numpy as np
import pandas as pd


class ModelService:
    """Service for managing ML model operations."""
    
    SUPPORTED_ALGORITHMS = {
        "random_forest": (RandomForestClassifier, RandomForestRegressor),
        "logistic_regression": (LogisticRegression, None),
        "linear_regression": (None, LinearRegression),
    }
    
    @staticmethod
    def train_model(
        df: pd.DataFrame,
        algorithm: str,
        target_column: str,
        test_split: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a machine learning model."""
        if algorithm not in ModelService.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
        
        # Train model
        clf, reg = ModelService.SUPPORTED_ALGORITHMS[algorithm]
        model_class = clf if y.dtype == 'object' else reg
        model = model_class(**kwargs) if model_class else None
        
        if model is None:
            raise ValueError(f"Cannot train {algorithm} for this target type")
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = ModelService.calculate_metrics(y_test, y_pred, algorithm)
        
        return {
            "model": model,
            "metrics": metrics,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, algorithm: str) -> Dict[str, float]:
        """Calculate performance metrics based on algorithm type."""
        if "regression" in algorithm.lower() or isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], (int, float)):
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2_score": float(r2_score(y_true, y_pred)),
            }
        else:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
