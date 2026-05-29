"""
stability_model.py — XGBoost stability classifier + regressor + SHAP explainer.

Models trained on DS1 (smart_grid_stability_augmented.csv).
Trained once via train() and then loaded from disk for inference.

Features used (from original 14 DS1 columns):
  tau1, tau2, tau3, tau4 — reaction times
  p1, p2, p3, p4         — power balance at each node
  g1, g2, g3, g4         — price elasticity

Targets:
  Classifier → stabf  ('stable' / 'unstable')
  Regressor  → stab   (continuous margin score)

Usage:
    from src.models.stability_model import StabilityModel
    model = StabilityModel()
    model.train()        # first time only
    result = model.predict(df_row)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from src.config import (
    DS1_AUGMENTED,
    STABILITY_CLASSIFIER_PATH,
    STABILITY_REGRESSOR_PATH,
    SCALER_DS1_PATH,
    MODELS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Features used for training (original 12 physics columns only)
FEATURE_COLS: List[str] = [
    "tau1", "tau2", "tau3", "tau4",
    "p1",   "p2",   "p3",   "p4",
    "g1",   "g2",   "g3",   "g4",
]
TARGET_CLASS = "stabf"
TARGET_REG   = "stab"


class StabilityModel:
    """
    Encapsulates the XGBoost stability classifier, regressor, SHAP explainer,
    and StandardScaler. All models are persisted to disk after training.
    """

    def __init__(self) -> None:
        self.classifier = None
        self.regressor  = None
        self.scaler     = None
        self._shap_explainer_cls = None  # SHAP TreeExplainer for classifier
        self._loaded    = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> None:
        """
        Train XGBoost classifier and regressor on DS1 and save to disk.

        Args:
            force: If True, re-train even if saved models already exist.
        """
        if (
            STABILITY_CLASSIFIER_PATH.exists()
            and STABILITY_REGRESSOR_PATH.exists()
            and not force
        ):
            logger.info("Stability models already trained — loading from disk.")
            self.load()
            return

        logger.info("Training XGBoost stability models on DS1...")

        # ── Load data ─────────────────────────────────────────────────────────
        df = pd.read_csv(DS1_AUGMENTED, usecols=FEATURE_COLS + [TARGET_CLASS, TARGET_REG])

        # Clean bracketed values e.g. '[6.38E-1]' → 0.638 that appear in some DS1 exports
        for col in FEATURE_COLS + [TARGET_REG]:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.strip("[]")
                .pipe(pd.to_numeric, errors="coerce")
            )
        df = df.dropna(subset=FEATURE_COLS + [TARGET_CLASS, TARGET_REG])

        X  = df[FEATURE_COLS]
        y_cls = (df[TARGET_CLASS] == "unstable").astype(int)  # 1=unstable, 0=stable
        y_reg = df[TARGET_REG]

        # ── Scale features ────────────────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ── Train/test split ──────────────────────────────────────────────────
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te = train_test_split(
            X_scaled, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
        )

        # ── XGBoost Classifier ────────────────────────────────────────────────
        from xgboost import XGBClassifier
        self.classifier = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.classifier.fit(X_tr, y_cls_tr)
        cls_acc = self.classifier.score(X_te, y_cls_te)
        logger.info(f"Classifier accuracy: {cls_acc:.4f}")

        # ── XGBoost Regressor ─────────────────────────────────────────────────
        from xgboost import XGBRegressor
        self.regressor = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        self.regressor.fit(X_tr, y_reg_tr)
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_reg_te, self.regressor.predict(X_te))
        logger.info(f"Regressor MAE: {mae:.6f}")

        # ── SHAP explainer ────────────────────────────────────────────────────
        import shap
        self._shap_explainer_cls = shap.TreeExplainer(self.classifier)

        # ── Save to disk ──────────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, STABILITY_CLASSIFIER_PATH)
        joblib.dump(self.regressor,  STABILITY_REGRESSOR_PATH)
        joblib.dump(self.scaler,     SCALER_DS1_PATH)
        logger.info("Stability models saved to disk.")
        self._loaded = True

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pre-trained models from disk."""
        if self._loaded:
            return
        self.classifier = joblib.load(STABILITY_CLASSIFIER_PATH)
        self.regressor  = joblib.load(STABILITY_REGRESSOR_PATH)
        self.scaler     = joblib.load(SCALER_DS1_PATH)

        import shap
        self._shap_explainer_cls = shap.TreeExplainer(self.classifier)
        self._loaded = True
        logger.info("Stability models loaded from disk.")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if STABILITY_CLASSIFIER_PATH.exists():
                self.load()
            else:
                raise RuntimeError(
                    "Stability models not trained yet. "
                    "Run StabilityModel().train() or scripts/setup_pipeline.py first."
                )

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, features: Dict) -> Dict:
        """
        Predict stability label, score, and SHAP feature importances.

        Args:
            features: Dict with keys matching FEATURE_COLS.

        Returns:
            {
              'label':       'stable' | 'unstable',
              'probability': float (0–1, probability of unstable),
              'stab_score':  float (continuous stability margin),
              'shap_top5':   [{feature, value, shap_value}, ...] (top 5 by |SHAP|),
              'health_score': int (0–100 grid health proxy),
            }
        """
        self._ensure_loaded()
        X = pd.DataFrame([{col: features.get(col, 0.0) for col in FEATURE_COLS}])
        X_scaled = self.scaler.transform(X)

        # Classifier
        prob_unstable = float(self.classifier.predict_proba(X_scaled)[0, 1])
        label = "unstable" if prob_unstable >= 0.5 else "stable"

        # Regressor
        stab_score = float(self.regressor.predict(X_scaled)[0])

        # SHAP
        shap_vals = self._shap_explainer_cls.shap_values(X_scaled)[0]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 (unstable)
        top5_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        shap_top5 = [
            {
                "feature":    FEATURE_COLS[i],
                "value":      float(X.iloc[0, i]),
                "shap_value": float(shap_vals[i]),
            }
            for i in top5_idx
        ]

        # Health score (0=worst, 100=best)
        # Maps prob_unstable ∈ [0,1] → health ∈ [100,0]
        health_score = max(0, min(100, int((1 - prob_unstable) * 100)))

        return {
            "label":        label,
            "probability":  round(prob_unstable, 4),
            "stab_score":   round(stab_score, 6),
            "shap_top5":    shap_top5,
            "health_score": health_score,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch inference on a DataFrame containing FEATURE_COLS.
        Returns the input DataFrame with added prediction columns.
        """
        self._ensure_loaded()
        X = df[FEATURE_COLS].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        df = df.copy()
        df["pred_prob_unstable"] = self.classifier.predict_proba(X_scaled)[:, 1]
        df["pred_label"]         = df["pred_prob_unstable"].apply(
            lambda p: "unstable" if p >= 0.5 else "stable"
        )
        df["pred_stab_score"]    = self.regressor.predict(X_scaled)
        df["pred_health_score"]  = ((1 - df["pred_prob_unstable"]) * 100).astype(int)
        return df


# ── Module-level singleton ─────────────────────────────────────────────────────
_stability_model: Optional[StabilityModel] = None


def get_stability_model() -> StabilityModel:
    """Return the shared StabilityModel singleton (auto-loads from disk)."""
    global _stability_model
    if _stability_model is None:
        _stability_model = StabilityModel()
    return _stability_model
