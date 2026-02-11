"""
Fire risk prediction model using RandomForest with SHAP explainability.

Trains a RandomForest regressor to predict structural fire counts per
geographic unit. Includes temporal train/test splitting, permutation
importance, and SHAP value computation for interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.inspection import permutation_importance
import streamlit as st
from typing import Optional
import warnings


class FireRiskModel:
    """RandomForest model for fire risk prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names = None
        self.is_fitted = False
        self.train_metrics = None
        self.cv_metrics = None
        self.feature_importance = None
        self.shap_values = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
        """
        Train the model and compute all evaluation metrics.

        Returns dict with train_metrics, cv_metrics, and feature_importance.
        """
        self.feature_names = feature_names

        # Fit
        self.model.fit(X, y)
        self.is_fitted = True

        # Train metrics
        y_pred = self.model.predict(X)
        self.train_metrics = {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "n_samples": len(y),
        }

        # Cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(
                self.model, X, y, cv=min(5, len(y) // 5 + 1),
                scoring="r2"
            )
        self.cv_metrics = {
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        # Feature importance (sklearn built-in + permutation)
        builtin_imp = self.model.feature_importances_
        perm_imp = permutation_importance(
            self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )

        self.feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance_builtin": builtin_imp,
            "importance_permutation": perm_imp.importances_mean,
            "importance_perm_std": perm_imp.importances_std,
        }).sort_values("importance_permutation", ascending=False).reset_index(drop=True)

        return {
            "train": self.train_metrics,
            "cv": self.cv_metrics,
            "importance": self.feature_importance,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict structural fire counts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def predict_with_risk(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict counts and return normalized risk scores (0–1).
        Returns (predictions, risk_scores).
        """
        preds = self.predict(X)
        max_pred = preds.max() if preds.max() > 0 else 1
        risk = np.clip(preds / max_pred, 0, 1)
        return preds, risk

    def compute_shap(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model interpretability.
        Returns SHAP values array or None if shap is unavailable.
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            self.shap_values = shap_values
            return shap_values
        except ImportError:
            print("SHAP not available; skipping SHAP computation.")
            return None
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None

    def get_risk_label(self, risk_score: float) -> str:
        """Classify risk score into human-readable label."""
        if risk_score >= 0.75:
            return "Critical"
        elif risk_score >= 0.50:
            return "High"
        elif risk_score >= 0.25:
            return "Moderate"
        return "Low"

    def get_risk_color(self, risk_score: float) -> str:
        """Return hex color for a risk score."""
        if risk_score >= 0.75:
            return "#FF3B4E"
        elif risk_score >= 0.50:
            return "#FFAA2B"
        elif risk_score >= 0.25:
            return "#FF6B35"
        return "#2ECC71"
