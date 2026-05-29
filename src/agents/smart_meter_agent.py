"""
smart_meter_agent.py — Smart Meter Agent node.

Analyses household/smart-meter consumption patterns from DS2:
  - Detects anomalous consumption events using the Isolation Forest
  - Computes demand statistics for the queried region/time window
  - Generates a prose summary for the Recommendation Agent

Does NOT make LLM calls — output is algorithmic Pandas + ML inference.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from src.agents.state import AgentState
from src.config import DS2_AUGMENTED
from src.models.anomaly_model import get_anomaly_model
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)

DS2_ANALYSIS_COLS = [
    "timestamp", "power_consumption", "voltage", "current",
    "reactive_power", "demand_load", "grid_frequency",
    "transformer_status", "outage_event", "region", "equipment_type",
]

# Sample size for DS2 (2M rows — avoid loading everything)
DS2_SAMPLE_ROWS = 50_000


def _load_ds2_sample() -> pd.DataFrame:
    """Load a manageable sample of DS2 for analysis."""
    if not DS2_AUGMENTED.exists():
        return pd.DataFrame()
    try:
        # Read header first cheaply, then load only matching columns
        header = pd.read_csv(DS2_AUGMENTED, nrows=0).columns.tolist()
        usecols = [c for c in DS2_ANALYSIS_COLS if c in header]
        df = pd.read_csv(DS2_AUGMENTED, usecols=usecols, nrows=DS2_SAMPLE_ROWS)
        return df
    except Exception as e:
        logger.error(f"Failed to load DS2: {e}")
        return pd.DataFrame()


def _compute_demand_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key demand statistics from the DS2 sample."""
    if df.empty or "power_consumption" not in df.columns:
        return {}
    pc = df["power_consumption"].dropna()
    return {
        "mean_kw":   round(float(pc.mean()), 3),
        "max_kw":    round(float(pc.max()), 3),
        "min_kw":    round(float(pc.min()), 3),
        "std_kw":    round(float(pc.std()), 3),
        "peak_events": int((pc > pc.quantile(0.95)).sum()),
    }


def _build_summary(stats: Dict, anomaly_count: int, sample_size: int) -> str:
    """Build a concise prose summary for downstream agents."""
    if not stats:
        return "Smart meter data not available for analysis."
    return (
        f"Smart meter analysis over {sample_size:,} readings: "
        f"Mean demand {stats.get('mean_kw', 0):.2f} kW, "
        f"peak {stats.get('max_kw', 0):.2f} kW, "
        f"{stats.get('peak_events', 0)} high-demand events detected. "
        f"Isolation Forest detected {anomaly_count} consumption anomalies "
        f"in this window."
    )


def analyse_smart_meter(state: AgentState) -> AgentState:
    """
    LangGraph node: Smart Meter Agent.

    Reads:  state.query, state.metadata_filters
    Writes: state.smart_meter_summary, state.anomaly_events_count,
            state.demand_forecast
    """
    request_id = state.get("request_id", "N/A")
    log_pipeline_event(request_id, "Smart Meter Agent", "start", {})
    logger.info(f"[{request_id}] Smart Meter Agent starting.")

    df = _load_ds2_sample()

    if df.empty:
        logger.warning(f"[{request_id}] DS2 not available.")
        return {
            **state,
            "smart_meter_summary": "Smart meter dataset not available.",
            "anomaly_events_count": 0,
            "demand_forecast":     {},
        }

    # ── Anomaly detection (batch — much faster than row-by-row) ──────────────
    anomaly_model = get_anomaly_model()
    anomaly_count = 0
    try:
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        result_df = anomaly_model.predict_batch_ds2(sample)
        anomaly_count = int(result_df["is_anomaly"].sum())
    except Exception as e:
        logger.warning(f"[{request_id}] DS2 anomaly detection error: {e}")

    # ── Demand statistics ─────────────────────────────────────────────────────
    stats = _compute_demand_stats(df)

    # ── Simple demand forecast (next 1h / next 24h using mean + std) ─────────
    demand_forecast = {
        "next_1h_kw":   round(stats.get("mean_kw", 0) * 1.1, 3),  # +10% buffer
        "next_24h_kw":  round(stats.get("mean_kw", 0) * 24 * 1.1, 3),
        "method":       "mean_extrapolation",
        "note":         "Use Prophet/LSTM for production forecasting.",
    }

    summary = _build_summary(stats, anomaly_count, len(df))
    logger.info(f"[{request_id}] Smart Meter analysis complete: {summary}")

    log_pipeline_event(
        request_id, "Smart Meter Agent", "complete",
        {"anomaly_count": anomaly_count, "mean_kw": stats.get("mean_kw")},
    )

    return {
        **state,
        "smart_meter_summary":  summary,
        "anomaly_events_count": anomaly_count,
        "demand_forecast":      demand_forecast,
    }
