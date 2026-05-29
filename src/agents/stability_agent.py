"""
stability_agent.py — Grid Stability Agent node.

Computes real-time stability scores, detects telemetry anomalies,
and generates SHAP feature explanations from DS1 data.

Uses XGBoost + Isolation Forest (no LLM calls — fully algorithmic).
Only the summary is passed downstream; raw DataFrames stay here.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from src.agents.state import AgentState
from src.config import DS1_AUGMENTED
from src.models.stability_model import get_stability_model
from src.models.anomaly_model import get_anomaly_model
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)

# Grid frequency normal band (Hz)
FREQ_NORMAL_MIN = 49.8
FREQ_NORMAL_MAX = 50.2


def _get_telemetry_window(
    region: Optional[str] = None,
    hours: int = 24,
    max_rows: int = 500,
) -> pd.DataFrame:
    """
    Load the most recent 'hours' worth of DS1 telemetry for the given region.
    Returns a DataFrame slice (max_rows rows to cap memory usage).
    """
    if not DS1_AUGMENTED.exists():
        return pd.DataFrame()

    # Read only required columns to reduce memory footprint
    cols = ["timestamp", "region", "equipment_type", "transformer_status",
            "outage_event", "stabf", "stab", "grid_frequency",
            "tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]

    df = pd.read_csv(DS1_AUGMENTED, usecols=[c for c in cols if c in
                     pd.read_csv(DS1_AUGMENTED, nrows=0).columns])

    if region and "region" in df.columns:
        df = df[df["region"] == region]

    # Take the last N rows as a proxy for "recent"
    return df.tail(max_rows).reset_index(drop=True)


def _detect_freq_status(freq_series: pd.Series) -> str:
    """Classify grid frequency status from a Series of Hz readings."""
    if freq_series.empty:
        return "unknown"
    mean_freq = freq_series.mean()
    if FREQ_NORMAL_MIN <= mean_freq <= FREQ_NORMAL_MAX:
        return "normal"
    elif abs(mean_freq - 50.0) > 0.5:
        return "excursion"
    else:
        return "deviation"


def assess_stability(state: AgentState) -> AgentState:
    """
    LangGraph node: Grid Stability Agent.

    Reads:  state.query, state.metadata_filters
    Writes: state.stability_label, state.stability_prob, state.stab_score,
            state.health_score, state.anomaly_flags, state.shap_top5,
            state.grid_freq_status
    """
    request_id       = state.get("request_id", "N/A")
    metadata_filters = state.get("metadata_filters", {})
    region           = metadata_filters.get("region")

    log_pipeline_event(request_id, "Grid Stability Agent", "start",
                       {"region": region})
    logger.info(f"[{request_id}] Grid Stability Agent — region={region}")

    # ── 1. Load telemetry window ──────────────────────────────────────────────
    df = _get_telemetry_window(region=region)

    if df.empty:
        logger.warning(f"[{request_id}] No telemetry data available for region={region}.")
        return {
            **state,
            "stability_label":  "unknown",
            "stability_prob":   None,
            "stab_score":       None,
            "health_score":     50,
            "anomaly_flags":    [],
            "shap_top5":        [],
            "grid_freq_status": "unknown",
        }

    # ── 2. Stability prediction on the most recent row ────────────────────────
    stability_model = get_stability_model()
    latest = df.iloc[-1].to_dict()
    pred = stability_model.predict(latest)

    logger.info(
        f"[{request_id}] Stability prediction: {pred['label']} "
        f"(prob={pred['probability']:.3f}, score={pred['stab_score']:.4f}, "
        f"health={pred['health_score']})"
    )

    # ── 3. Anomaly detection across the window ────────────────────────────────
    anomaly_model = get_anomaly_model()
    anomaly_df    = anomaly_model.detect_batch_ds1(df, window_region=region)

    anomaly_flags: List[Dict[str, Any]] = []
    for _, row in anomaly_df.head(10).iterrows():
        anomaly_flags.append({
            "timestamp":         str(row.get("timestamp", "N/A")),
            "equipment_type":    str(row.get("equipment_type", "N/A")),
            "transformer_status": str(row.get("transformer_status", "N/A")),
            "stab":              float(row.get("stab", 0)),
            "anomaly_score":     float(row.get("anomaly_score", 0)),
        })

    logger.info(f"[{request_id}] Anomaly detection: {len(anomaly_flags)} anomalies found.")

    # ── 4. Grid frequency status ──────────────────────────────────────────────
    freq_col = df["grid_frequency"] if "grid_frequency" in df.columns else pd.Series([50.0])
    grid_freq_status = _detect_freq_status(freq_col)

    log_pipeline_event(
        request_id, "Grid Stability Agent", "complete",
        {
            "stability_label":   pred["label"],
            "health_score":      pred["health_score"],
            "anomaly_count":     len(anomaly_flags),
            "grid_freq_status":  grid_freq_status,
        },
    )

    return {
        **state,
        "stability_label":  pred["label"],
        "stability_prob":   pred["probability"],
        "stab_score":       pred["stab_score"],
        "health_score":     pred["health_score"],
        "anomaly_flags":    anomaly_flags,
        "shap_top5":        pred["shap_top5"],
        "grid_freq_status": grid_freq_status,
    }
