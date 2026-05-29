"""
anomaly_model.py — Isolation Forest anomaly detectors for DS1 and DS2.

DS1 Anomaly Detector:
  Detects out-of-distribution grid states from stability telemetry.
  Features: tau1–tau4, p1–p4, g1–g4 (same as stability model).

DS2 Anomaly Detector:
  Detects abnormal consumption patterns from smart meter data.
  Features: power_consumption, voltage, current, reactive_power, demand_load.

Usage:
    from src.models.anomaly_model import get_anomaly_model
    model = get_anomaly_model()
    model.train()
    result = model.predict_ds1(feature_dict)
    result = model.predict_ds2(feature_dict)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.config import (
    DS1_AUGMENTED, DS2_AUGMENTED,
    ANOMALY_DS1_PATH, ANOMALY_DS2_PATH,
    SCALER_DS2_PATH, MODELS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

DS1_FEATURES = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
DS2_FEATURES = ["power_consumption", "voltage", "current", "reactive_power", "demand_load"]


class AnomalyModel:
    """Isolation Forest anomaly detectors for DS1 and DS2."""

    def __init__(self) -> None:
        self._model_ds1  = None
        self._model_ds2  = None
        self._scaler_ds2 = None
        self._loaded     = False

    def train(self, force: bool = False) -> None:
        """Train Isolation Forest on DS1 and DS2 and save to disk."""
        if ANOMALY_DS1_PATH.exists() and ANOMALY_DS2_PATH.exists() and not force:
            logger.info("Anomaly models already trained — loading from disk.")
            self.load()
            return

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # ── DS1 Anomaly Detector ──────────────────────────────────────────────
        if DS1_AUGMENTED.exists():
            ds1 = pd.read_csv(DS1_AUGMENTED, usecols=DS1_FEATURES)
            self._model_ds1 = IsolationForest(
                n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
            )
            self._model_ds1.fit(ds1[DS1_FEATURES])
            joblib.dump(self._model_ds1, ANOMALY_DS1_PATH)
            logger.info(f"DS1 anomaly model trained on {len(ds1)} rows.")
        else:
            logger.warning("DS1 augmented file not found — skipping DS1 anomaly model.")

        # ── DS2 Anomaly Detector ──────────────────────────────────────────────
        if DS2_AUGMENTED.exists():
            # DS2 is large — sample 100K rows for training
            ds2 = pd.read_csv(
                DS2_AUGMENTED,
                usecols=DS2_FEATURES,
                nrows=100_000,
            ).dropna()
            self._scaler_ds2 = StandardScaler()
            X2 = self._scaler_ds2.fit_transform(ds2[DS2_FEATURES])
            self._model_ds2 = IsolationForest(
                n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
            )
            self._model_ds2.fit(X2)
            joblib.dump(self._model_ds2,  ANOMALY_DS2_PATH)
            joblib.dump(self._scaler_ds2, SCALER_DS2_PATH)
            logger.info(f"DS2 anomaly model trained on {len(ds2)} rows.")
        else:
            logger.warning("DS2 augmented file not found — skipping DS2 anomaly model.")

        self._loaded = True

    def load(self) -> None:
        """Load pre-trained models from disk."""
        if self._loaded:
            return
        if ANOMALY_DS1_PATH.exists():
            self._model_ds1 = joblib.load(ANOMALY_DS1_PATH)
        if ANOMALY_DS2_PATH.exists():
            self._model_ds2  = joblib.load(ANOMALY_DS2_PATH)
            self._scaler_ds2 = joblib.load(SCALER_DS2_PATH)
        self._loaded = True
        logger.info("Anomaly models loaded from disk.")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if ANOMALY_DS1_PATH.exists() or ANOMALY_DS2_PATH.exists():
                self.load()
            else:
                raise RuntimeError("Anomaly models not trained. Run setup_pipeline.py first.")

    def predict_ds1(self, features: Dict) -> Dict:
        """
        Predict if a grid telemetry reading is anomalous (DS1 feature space).

        Returns:
            {
              'is_anomaly': bool,
              'anomaly_score': float (lower = more anomalous; < 0 = anomaly),
            }
        """
        self._ensure_loaded()
        if self._model_ds1 is None:
            return {"is_anomaly": False, "anomaly_score": 0.0}

        X = pd.DataFrame([{col: features.get(col, 0.0) for col in DS1_FEATURES}])
        score = float(self._model_ds1.decision_function(X)[0])
        label = int(self._model_ds1.predict(X)[0])  # -1=anomaly, 1=normal
        return {
            "is_anomaly":    label == -1,
            "anomaly_score": round(score, 4),
        }

    def predict_batch_ds2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch anomaly detection on DS2 smart meter data.
        Returns df with added 'anomaly_score' and 'is_anomaly' columns.
        """
        self._ensure_loaded()
        if self._model_ds2 is None or self._scaler_ds2 is None:
            df = df.copy()
            df["anomaly_score"] = 0.0
            df["is_anomaly"] = False
            return df

        available = [c for c in DS2_FEATURES if c in df.columns]
        X = df[available].fillna(0.0)
        X_scaled = self._scaler_ds2.transform(X)
        scores = self._model_ds2.decision_function(X_scaled)
        labels = self._model_ds2.predict(X_scaled)

        df = df.copy()
        df["anomaly_score"] = scores
        df["is_anomaly"] = labels == -1
        return df

    def predict_ds2(self, features: Dict) -> Dict:
        """
        Predict if a smart meter reading is anomalous (DS2 feature space).

        Returns:
            {
              'is_anomaly': bool,
              'anomaly_score': float,
            }
        """
        self._ensure_loaded()
        if self._model_ds2 is None or self._scaler_ds2 is None:
            return {"is_anomaly": False, "anomaly_score": 0.0}

        X = pd.DataFrame([{col: features.get(col, 0.0) for col in DS2_FEATURES}])
        X_scaled = self._scaler_ds2.transform(X)
        score = float(self._model_ds2.decision_function(X_scaled)[0])
        label = int(self._model_ds2.predict(X_scaled)[0])
        return {
            "is_anomaly":    label == -1,
            "anomaly_score": round(score, 4),
        }

    def detect_batch_ds1(self, df: pd.DataFrame, window_region: Optional[str] = None) -> pd.DataFrame:
        """
        Batch anomaly detection on DS1 telemetry.
        Optionally filter to a specific region before analysis.

        Returns subset of df where anomalies were detected, with anomaly_score column.
        """
        self._ensure_loaded()
        if self._model_ds1 is None:
            return pd.DataFrame()

        if window_region and "region" in df.columns:
            df = df[df["region"] == window_region].copy()

        available = [c for c in DS1_FEATURES if c in df.columns]
        X = df[available].fillna(0.0)
        scores = self._model_ds1.decision_function(X)
        labels = self._model_ds1.predict(X)

        df = df.copy()
        df["anomaly_score"] = scores
        df["is_anomaly"]    = labels == -1
        return df[df["is_anomaly"]].sort_values("anomaly_score")


# ── Module-level singleton ─────────────────────────────────────────────────────
_anomaly_model: Optional[AnomalyModel] = None


def get_anomaly_model() -> AnomalyModel:
    """Return the shared AnomalyModel singleton."""
    global _anomaly_model
    if _anomaly_model is None:
        _anomaly_model = AnomalyModel()
    return _anomaly_model
