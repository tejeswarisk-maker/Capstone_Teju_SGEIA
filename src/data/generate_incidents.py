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
        "Consumption peaked at {power_consumption:.3f}kW (reactive: {reactive_power:.3f}kVAR) — "
        "exceeding 85th percentile threshold. "
        "Demand load: {demand_load:.3f}kW. Voltage: {voltage:.1f}V. Current: {current:.1f}A. "
        "Sub-metering: kitchen {sub_metering_kitchen:.1f}W, "
        "laundry {sub_metering_laundry:.1f}W, HVAC {sub_metering_hvac:.1f}W. "
        "Grid frequency: {grid_frequency:.3f}Hz. Transformer: {transformer_status}. "
        "Load shedding protocols activated to prevent cascade failure."
    ),
    "meter_dropout": (
        "Smart meter communication loss detected at {equipment_type} in {region}. "
        "Last recorded consumption: {power_consumption:.3f}kW (reactive: {reactive_power:.3f}kVAR). "
        "Voltage at dropout: {voltage:.1f}V, current: {current:.1f}A. "
        "Sub-metering breakdown — kitchen: {sub_metering_kitchen:.1f}W, "
        "laundry: {sub_metering_laundry:.1f}W, HVAC: {sub_metering_hvac:.1f}W. "
        "Grid frequency: {grid_frequency:.3f}Hz. Transformer status: {transformer_status}. "
        "Field team dispatched for meter hardware inspection."
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


# Event types sourced from DS1 (grid stability telemetry)
DS1_EVENT_TYPES = {
    "frequency_excursion", "partial_outage", "full_outage",
    "voltage_deviation", "renewable_variability", "no_event",
}

# Event types sourced from DS2 (smart meter consumption data)
DS2_EVENT_TYPES = {
    "meter_dropout", "high_demand_event",
}


def _seed_from_ds1(ds1: pd.DataFrame, event_type: str, count: int,
                   rng: np.random.Generator) -> List[Dict]:
    """
    Build incident seed rows for grid-stability event types using DS1.
    DS1 columns used: grid_frequency, p1, g1, tau1, outage_event
    Voltage/current/power are derived from real DS1 telemetry values.
    """
    subset = ds1[ds1["outage_event"] == event_type]
    if len(subset) == 0:
        subset = ds1  # fallback to full DS1

    sampled = subset.sample(n=min(count, len(subset)), random_state=42, replace=True)
    rows = []

    for _, row in sampled.iterrows():
        region    = rng.choice(REGIONS)
        equipment = rng.choice(EQUIPMENT_TYPES)

        # Derive realistic voltage from DS1 grid_frequency (correlated in real grids)
        freq    = float(row.get("grid_frequency", 50.0))
        voltage = freq * 4.8 + rng.normal(0, 4)
        voltage = round(max(220.0, min(260.0, voltage)), 2)

        # Derive current from DS1 p1 (power at node 1 — real physics value)
        current  = round(abs(float(row.get("p1", 2.5))) * 8 + rng.uniform(2, 12), 2)
        power_kw = round(current * voltage / 1000, 2)
        demand   = round(power_kw * rng.uniform(0.88, 1.12), 2)

        rows.append({
            "event_type":         event_type,
            "region":             region,
            "equipment_type":     equipment,
            "severity":           SEVERITY_MAP[event_type],
            "transformer_status": TRANSFORMER_STATUS_MAP[event_type],
            "voltage":            voltage,
            "current":            current,
            "power_consumption":  power_kw,
            "demand_load":        demand,
            "grid_frequency":     round(freq, 3),
            # DS1-specific extras included in description context
            "tau1":               round(float(row.get("tau1", 4.0)), 4),
            "p1":                 round(float(row.get("p1", 3.5)), 4),
            "stab_score":         round(float(row.get("stab", 0.0)), 6),
            "data_source":        "DS1",
        })

    return rows


def _seed_from_ds2(ds2: pd.DataFrame, event_type: str, count: int,
                   rng: np.random.Generator) -> List[Dict]:
    """
    Build incident seed rows for smart-meter event types using DS2.
    DS2 columns used: power_consumption, reactive_power, voltage, current,
                      demand_load, grid_frequency, sub_metering_kitchen/laundry/hvac

    This correctly grounds meter_dropout and high_demand_event incidents
    in REAL household consumption patterns, not in DS1 physics values.
    """
    # For high_demand_event: sample rows where demand is in top 15%
    # For meter_dropout: sample rows with unusual consumption patterns
    if event_type == "high_demand_event":
        threshold = ds2["power_consumption"].quantile(0.85)
        subset = ds2[ds2["power_consumption"] >= threshold]
    elif event_type == "meter_dropout":
        # Dropout = sudden anomalous reading — very low or very high consumption
        lo = ds2["power_consumption"].quantile(0.02)
        hi = ds2["power_consumption"].quantile(0.97)
        subset = ds2[
            (ds2["power_consumption"] <= lo) | (ds2["power_consumption"] >= hi)
        ]
    else:
        subset = ds2

    if len(subset) == 0:
        subset = ds2

    sampled = subset.sample(n=min(count, len(subset)), random_state=42, replace=True)
    rows = []

    # DS2 zones are Zone_A_Residential etc — map to project zone names
    zone_map = {
        "Zone_A_Residential": "Zone_A", "Zone_B_Residential": "Zone_B",
        "Zone_C_Residential": "Zone_C", "Zone_D_Residential": "Zone_D",
    }
    sm_equipment = ["smart_meter_bank", "distribution_unit", "substation"]

    for _, row in sampled.iterrows():
        raw_region = str(row.get("region", "Zone_A_Residential"))
        region     = zone_map.get(raw_region, rng.choice(REGIONS))
        equipment  = rng.choice(sm_equipment)

        # Use REAL DS2 consumption values — no derivation from physics
        power_kw  = round(float(row.get("power_consumption", 1.5)), 3)
        reactive  = round(float(row.get("reactive_power",   0.1)), 3)
        voltage   = round(float(row.get("voltage",        240.0)), 2)
        current   = round(float(row.get("current",          6.0)), 2)
        demand    = round(float(row.get("demand_load",  power_kw)), 3)
        freq      = round(float(row.get("grid_frequency",  50.0)), 3)
        sm_kitch  = round(float(row.get("sub_metering_kitchen", 0.0)), 1)
        sm_laund  = round(float(row.get("sub_metering_laundry", 0.0)), 1)
        sm_hvac   = round(float(row.get("sub_metering_hvac",    0.0)), 1)

        rows.append({
            "event_type":           event_type,
            "region":               region,
            "equipment_type":       equipment,
            "severity":             SEVERITY_MAP[event_type],
            "transformer_status":   TRANSFORMER_STATUS_MAP[event_type],
            "voltage":              voltage,
            "current":              current,
            "power_consumption":    power_kw,
            "reactive_power":       reactive,
            "demand_load":          demand,
            "grid_frequency":       freq,
            "sub_metering_kitchen": sm_kitch,
            "sub_metering_laundry": sm_laund,
            "sub_metering_hvac":    sm_hvac,
            "data_source":          "DS2",
        })

    return rows


def _build_seed_rows(ds1: pd.DataFrame, ds2: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """
    Build 200 incident seed rows with correct data sourcing:
      - Grid stability events (full_outage, partial_outage, frequency_excursion,
        voltage_deviation, renewable_variability, no_event) → DS1
      - Smart meter events (meter_dropout, high_demand_event) → DS2

    This ensures every incident's telemetry values match the dataset
    the corresponding ML model was trained on.
    """
    rng  = np.random.default_rng(seed=99)
    rows: List[Dict] = []

    for event_type, count in OUTAGE_DIST.items():
        if event_type in DS1_EVENT_TYPES:
            rows.extend(_seed_from_ds1(ds1, event_type, count, rng))
            logger.info(f"  DS1 seed: {event_type} × {count}")
        elif event_type in DS2_EVENT_TYPES:
            rows.extend(_seed_from_ds2(ds2, event_type, count, rng))
            logger.info(f"  DS2 seed: {event_type} × {count}")

    rng.shuffle(rows)
    return pd.DataFrame(rows[:n])


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
    """Fill a deterministic template from the incident seed data.
    DS2-sourced rows have extra fields (reactive_power, sub_metering_*) that
    DS1 rows don't have — provide safe defaults so format() never raises KeyError.
    """
    template = TEMPLATES.get(row["event_type"], TEMPLATES["no_event"])
    safe_row = {
        # defaults for DS2-only fields
        "reactive_power":       0.0,
        "sub_metering_kitchen": 0.0,
        "sub_metering_laundry": 0.0,
        "sub_metering_hvac":    0.0,
        # defaults for DS1-only fields
        "tau1":       4.0,
        "p1":         3.5,
        "stab_score": 0.0,
        **row,  # actual row values override defaults
    }
    return template.format(**safe_row)


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

    # ── 1. Load DS1 (grid stability telemetry) ────────────────────────────────
    if DS1_AUGMENTED.exists():
        ds1 = pd.read_csv(DS1_AUGMENTED)
        logger.info(f"DS1 loaded: {len(ds1)} rows for grid incident seeding.")
    else:
        logger.warning("DS1 not found — grid incidents will use fallback values.")
        ds1 = pd.DataFrame()

    # ── 2. Load DS2 (smart meter consumption data) ────────────────────────────
    from src.config import DS2_AUGMENTED
    if DS2_AUGMENTED.exists():
        # DS2 is 2M rows — sample a representative chunk for seeding
        ds2 = pd.read_csv(
            DS2_AUGMENTED,
            usecols=["power_consumption", "reactive_power", "voltage", "current",
                     "demand_load", "grid_frequency", "region",
                     "sub_metering_kitchen", "sub_metering_laundry", "sub_metering_hvac"],
            nrows=200_000,
        ).dropna(subset=["power_consumption", "voltage", "current"])
        logger.info(f"DS2 loaded: {len(ds2)} rows for smart meter incident seeding.")
    else:
        logger.warning("DS2 not found — meter incidents will use fallback values.")
        ds2 = pd.DataFrame()

    # ── 3. Build seed rows — DS1 for grid events, DS2 for meter events ────────
    seed_df = _build_seed_rows(ds1, ds2, n=200)
    logger.info(f"Seed rows: {len(seed_df)} | DS1 events: {seed_df[seed_df['data_source']=='DS1'].shape[0] if 'data_source' in seed_df.columns else '?'} | DS2 events: {seed_df[seed_df['data_source']=='DS2'].shape[0] if 'data_source' in seed_df.columns else '?'}")
    logger.info(f"Seed rows built: {len(seed_df)} incidents across {seed_df['event_type'].nunique()} event types")

    # ── 3. Optionally initialise OpenAI client ────────────────────────────────
    client = None
    if use_llm and settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
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
