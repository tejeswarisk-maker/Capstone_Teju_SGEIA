"""
generate_incidents.py — Generate the synthetic grid incidents RAG corpus.

Output: data/grid_incidents_synthetic.csv  (200 rows × 13 cols)

This file is the PRIMARY RAG document store. Each row's 'description' field
contains a natural-language incident narrative that gets embedded into ChromaDB
for semantic retrieval.

Strategy:
  1. Load augmented DS1 to sample realistic telemetry seed values.
  2. Define incident templates covering all outage types / severities.
  3. Use OpenAI gpt-4o-mini to generate varied narrative text for each row.
  4. Fall back to deterministic template fill-in if the API call fails.

Run:
    python -m src.data.generate_incidents
"""

import random
import time
import uuid
from typing import Dict, List

import pandas as pd
import numpy as np

from src.config import (
    DS1_AUGMENTED, INCIDENTS_CSV, settings
)
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)
REQUEST_ID = "generate_incidents"

# ── Target distribution for 200 incidents ──────────────────────────────────────
OUTAGE_DIST = {
    "frequency_excursion": 27,
    "partial_outage":       25,
    "high_demand_event":    25,
    "voltage_deviation":    25,
    "meter_dropout":        20,
    "no_event":             18,
    "full_outage":          30,
    "renewable_variability": 30,
}

SEVERITY_MAP = {
    "no_event":             "low",
    "voltage_deviation":    "medium",
    "meter_dropout":        "medium",
    "high_demand_event":    "high",
    "frequency_excursion":  "high",
    "partial_outage":       "high",
    "full_outage":          "critical",
    "renewable_variability": "medium",
}

EQUIPMENT_TYPES = [
    "transformer", "distribution_unit", "substation",
    "smart_meter_bank", "transmission_line", "renewable_inverter",
]

REGIONS = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]

TRANSFORMER_STATUS_MAP = {
    "no_event":             "normal",
    "voltage_deviation":    "monitoring",
    "meter_dropout":        "monitoring",
    "high_demand_event":    "overloaded",
    "frequency_excursion":  "degraded",
    "partial_outage":       "overloaded",
    "full_outage":          "critical",
    "renewable_variability": "degraded",
}

# ── Narrative templates (fallback when LLM is unavailable) ────────────────────
TEMPLATES: Dict[str, str] = {
    "full_outage": (
        "Full power outage reported affecting {equipment_type} cluster in {region}. "
        "Approximately 80-100% of connected load lost. Voltage collapsed to {voltage:.1f}V. "
        "Peak demand prior to event: {power_consumption:.2f}kW. "
        "Grid frequency deviated to {grid_frequency:.2f}Hz. "
        "Transformer status: {transformer_status}. Restoration team dispatched."
    ),
    "partial_outage": (
        "Partial outage reported affecting {equipment_type} cluster in {region}. "
        "Approx. 30-40% of connected load lost. Voltage collapsed to {voltage:.1f}V. "
        "Peak demand prior to event: {power_consumption:.2f}kW. "
        "Grid frequency deviated to {grid_frequency:.2f}Hz. "
        "Transformer status: {transformer_status}. Restoration in progress."
    ),
    "voltage_deviation": (
        "Voltage deviation detected on {equipment_type} in {region}. "
        "Measured voltage: {voltage:.1f}V (nominal: 240V). "
        "Current draw: {current:.1f}A. "
        "Grid frequency: {grid_frequency:.2f}Hz. "
        "Demand load at time of event: {demand_load:.2f}kW. "
        "Transformer status: {transformer_status}. Engineers investigating root cause."
    ),
    "frequency_excursion": (
        "Grid frequency excursion detected in {region}. "
        "Frequency measured at {grid_frequency:.2f}Hz (nominal: 50Hz). "
        "Affected equipment: {equipment_type}. Voltage: {voltage:.1f}V. "
        "Power consumption at event: {power_consumption:.2f}kW. "
        "Transformer status: {transformer_status}. "
        "Automatic frequency regulation engaged. Monitoring for reoccurrence."
    ),
    "high_demand_event": (
        "High demand event recorded at {equipment_type} in {region}. "
        "Demand load reached {demand_load:.2f}kW — exceeding 80% of rated capacity. "
        "Voltage: {voltage:.1f}V. Current: {current:.1f}A. "
        "Grid frequency: {grid_frequency:.2f}Hz. "
        "Transformer status: {transformer_status}. "
        "Load shedding protocols activated to prevent cascade failure."
    ),
    "meter_dropout": (
        "Smart meter communication loss detected in {region}. "
        "Equipment type: {equipment_type}. "
        "Last recorded voltage: {voltage:.1f}V. "
        "Last recorded power consumption: {power_consumption:.2f}kW. "
        "Grid frequency at time of dropout: {grid_frequency:.2f}Hz. "
        "Transformer status: {transformer_status}. "
        "Field team dispatched for meter inspection."
    ),
    "no_event": (
        "Routine monitoring report for {equipment_type} in {region}. "
        "All parameters within normal operating range. "
        "Voltage: {voltage:.1f}V. Current: {current:.1f}A. "
        "Power consumption: {power_consumption:.2f}kW. "
        "Grid frequency: {grid_frequency:.2f}Hz. "
        "Transformer status: {transformer_status}. No action required."
    ),
    "renewable_variability": (
        "Renewable energy variability event detected in {region}. "
        "Intermittent generation fluctuations at {equipment_type} causing instability. "
        "Voltage swing: {voltage:.1f}V. Grid frequency deviation: {grid_frequency:.2f}Hz. "
        "Power consumption during event: {power_consumption:.2f}kW. "
        "Transformer status: {transformer_status}. "
        "Grid balancing reserves activated. Storage dispatch initiated."
    ),
}

LLM_SYSTEM_PROMPT = """You are a utility grid operations expert writing incident reports for a grid monitoring system.
Write a natural-language incident description for the provided grid event data.
The description should be 2-4 sentences, technically accurate, and include all the provided metrics.
Write only the description text — no labels, no JSON, no headers."""


def _build_seed_rows(ds1: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """
    Sample realistic telemetry values from DS1 to seed the incident records.
    Stratify by outage_event to match the target distribution.
    """
    rng = np.random.default_rng(seed=99)
    rows: List[Dict] = []

    for event_type, count in OUTAGE_DIST.items():
        subset = ds1[ds1["outage_event"] == event_type]
        if len(subset) == 0:
            # Fall back to random rows if this event type not found in DS1
            subset = ds1

        sampled = subset.sample(n=min(count, len(subset)), random_state=42, replace=True)

        for _, row in sampled.iterrows():
            region    = rng.choice(REGIONS)
            equipment = rng.choice(EQUIPMENT_TYPES)
            severity  = SEVERITY_MAP[event_type]
            xfmr_stat = TRANSFORMER_STATUS_MAP[event_type]

            # Add slight noise to telemetry values for variety
            voltage   = float(row.get("grid_frequency", 50.0)) * 4.8 + rng.normal(0, 5)
            voltage   = max(200.0, min(260.0, voltage))  # clamp to realistic range
            current   = abs(float(row.get("p1", 2.5))) * 8 + rng.uniform(2, 15)
            power_kw  = current * voltage / 1000
            demand_kw = power_kw * rng.uniform(0.9, 1.1)
            freq      = float(row.get("grid_frequency", 50.0)) if "grid_frequency" in row else (
                50.0 + rng.normal(0, 0.3)
            )

            rows.append({
                "event_type":        event_type,
                "region":            region,
                "equipment_type":    equipment,
                "severity":          severity,
                "transformer_status": xfmr_stat,
                "voltage":           round(voltage, 2),
                "current":           round(current, 2),
                "power_consumption": round(power_kw, 2),
                "demand_load":       round(demand_kw, 2),
                "grid_frequency":    round(freq, 3),
            })

    return pd.DataFrame(rows[:n])  # trim to exactly n records


def _generate_description_llm(row: Dict, client) -> str:
    """
    Use gpt-4o-mini to generate a varied incident description.
    Returns the LLM output or falls back to the template on failure.
    """
    user_message = (
        f"Event type: {row['event_type']}\n"
        f"Region: {row['region']}\n"
        f"Equipment: {row['equipment_type']}\n"
        f"Severity: {row['severity']}\n"
        f"Voltage: {row['voltage']}V\n"
        f"Current: {row['current']}A\n"
        f"Power consumption: {row['power_consumption']}kW\n"
        f"Demand load: {row['demand_load']}kW\n"
        f"Grid frequency: {row['grid_frequency']}Hz\n"
        f"Transformer status: {row['transformer_status']}"
    )
    try:
        response = client.chat.completions.create(
            model=settings.openai_model_simple,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM description generation failed: {e} — using template fallback.")
        return _generate_description_template(row)


def _generate_description_template(row: Dict) -> str:
    """Fill a deterministic template from the incident seed data."""
    template = TEMPLATES.get(row["event_type"], TEMPLATES["no_event"])
    return template.format(**row)


def run_generation(force: bool = False, use_llm: bool = True) -> pd.DataFrame:
    """
    Generate the synthetic incident corpus.

    Args:
        force:   Re-generate even if the output file already exists.
        use_llm: If True, use OpenAI gpt-4o-mini for richer descriptions.
                 Falls back to templates on API error.

    Returns:
        The incidents DataFrame (200 rows × 13 cols).
    """
    if INCIDENTS_CSV.exists() and not force:
        logger.info("Incidents CSV already exists — loading from disk.")
        return pd.read_csv(INCIDENTS_CSV)

    logger.info("Starting synthetic incident generation.")
    log_pipeline_event(REQUEST_ID, "Incident Generation", "start", {"n_incidents": 200})

    # ── 1. Load augmented DS1 for telemetry seeds ──────────────────────────────
    if DS1_AUGMENTED.exists():
        ds1 = pd.read_csv(DS1_AUGMENTED)
    else:
        logger.warning("DS1 augmented file not found — using random seed values.")
        ds1 = pd.DataFrame()

    # ── 2. Build seed rows ─────────────────────────────────────────────────────
    seed_df = _build_seed_rows(ds1, n=200)
    logger.info(f"Seed rows built: {len(seed_df)} incidents across {seed_df['event_type'].nunique()} event types")

    # ── 3. Optionally initialise OpenAI client ────────────────────────────────
    client = None
    if use_llm and settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialised for description generation.")
        except ImportError:
            logger.warning("openai package not available — falling back to templates.")

    # ── 4. Generate descriptions ───────────────────────────────────────────────
    records = []
    for i, (_, row) in enumerate(seed_df.iterrows()):
        row_dict = row.to_dict()

        if client:
            description = _generate_description_llm(row_dict, client)
            time.sleep(0.05)  # gentle rate limiting
        else:
            description = _generate_description_template(row_dict)

        records.append({
            "incident_id":       f"INC-{i + 1:04d}",
            "timestamp":         pd.Timestamp("2021-01-01") + pd.Timedelta(days=random.randint(0, 1095)),
            "region":            row_dict["region"],
            "equipment_type":    row_dict["equipment_type"],
            "voltage":           row_dict["voltage"],
            "current":           row_dict["current"],
            "power_consumption": row_dict["power_consumption"],
            "demand_load":       row_dict["demand_load"],
            "grid_frequency":    row_dict["grid_frequency"],
            "transformer_status": row_dict["transformer_status"],
            "outage_event":      row_dict["event_type"],
            "severity":          row_dict["severity"],
            "description":       description,
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  Generated {i + 1}/200 incidents...")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    INCIDENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INCIDENTS_CSV, index=False)
    logger.info(f"Incidents CSV saved: {INCIDENTS_CSV} ({len(df)} rows)")

    log_pipeline_event(
        REQUEST_ID, "Incident Generation", "complete",
        {"rows": len(df), "event_types": df["outage_event"].value_counts().to_dict()},
    )
    return df


if __name__ == "__main__":
    run_generation(force=False, use_llm=True)
