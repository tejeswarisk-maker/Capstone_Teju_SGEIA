"""
augment_ds2.py — Augment the Household Power Consumption dataset (DS2).

Input : data/source_files/household_power_consumption.txt  (9 cols, ~2.07M rows)
Output: data/household_power_consumption.csv               (16 cols, ~2.07M rows)
Backup: data/originals/household_power_consumption_ORIGINAL.txt

Changes made:
  Renamed columns   — all 9 original columns renamed to key-field standard names
  Added columns (7) — timestamp, demand_load, grid_frequency, region,
                      equipment_type, transformer_status, outage_event

Run:
    python -m src.data.augment_ds2
"""

import shutil
import numpy as np
import pandas as pd

from src.config import DS2_AUGMENTED, DS2_ORIGINAL, ORIGINALS_DIR
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)
REQUEST_ID = "augment_ds2"

# Column rename map: original → standard key field name
COLUMN_RENAME = {
    "Date":                    "date",
    "Time":                    "time",
    "Global_active_power":     "power_consumption",
    "Global_reactive_power":   "reactive_power",
    "Voltage":                 "voltage",
    "Global_intensity":        "current",
    "Sub_metering_1":          "sub_metering_kitchen",
    "Sub_metering_2":          "sub_metering_laundry",
    "Sub_metering_3":          "sub_metering_hvac",
}


def _derive_transformer_status(voltage: pd.Series) -> pd.Series:
    """
    Map voltage band to transformer operational status.
      < 230V   → degraded   (under-voltage)
      230–237V → monitoring (low but acceptable)
      237–245V → normal     (nominal band)
      > 245V   → monitoring (over-voltage)
    """
    status = pd.Series("normal", index=voltage.index)
    status[voltage < 230]  = "degraded"
    status[(voltage >= 230) & (voltage < 237)] = "monitoring"
    status[voltage > 245]  = "monitoring"
    return status


def _derive_outage_event(df: pd.DataFrame) -> pd.Series:
    """
    Derive outage event from missing power and high-demand flags.
      Missing power (NaN after imputation was skipped) → meter_dropout
      power_consumption > 8.0 kW                      → high_demand_event
      Otherwise                                        → no_event
    """
    # We use the raw column before forward-fill; NaN rows = meter dropout
    events = pd.Series("no_event", index=df.index)
    events[df["_is_missing"]]               = "meter_dropout"
    events[df["power_consumption"] > 8.0]   = "high_demand_event"
    return events


def run_augmentation(force: bool = False, chunksize: int = 200_000) -> None:
    """
    Run the DS2 augmentation pipeline.

    Due to the 2M+ row size, the file is processed in chunks and written
    to the output CSV incrementally to avoid memory errors.

    Args:
        force:     If True, re-augment even if the output file already exists.
        chunksize: Number of rows to process per chunk (default 200K).
    """
    if DS2_AUGMENTED.exists() and not force:
        logger.info("DS2 augmented file already exists — skipping.")
        return

    logger.info("Starting DS2 augmentation pipeline (chunked processing).")
    log_pipeline_event(
        REQUEST_ID, "DS2 Augmentation", "start",
        {"source": str(DS2_ORIGINAL), "output": str(DS2_AUGMENTED)},
    )

    # ── 1. Back up original ───────────────────────────────────────────────────
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = ORIGINALS_DIR / "household_power_consumption_ORIGINAL.txt"
    if not backup_path.exists():
        shutil.copy(DS2_ORIGINAL, backup_path)
        logger.info(f"Backup written to {backup_path}")

    # ── 2. Detect voltage nominal for frequency derivation ────────────────────
    # Sample first 10K rows to compute nominal voltage
    sample = pd.read_csv(
        DS2_ORIGINAL, sep=";", nrows=10_000, na_values=["?"],
        usecols=["Voltage"],
    )
    voltage_nominal = sample["Voltage"].mean()
    logger.info(f"Voltage nominal (sample mean): {voltage_nominal:.2f} V")

    rng = np.random.default_rng(seed=42)
    DS2_AUGMENTED.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    first_chunk = True

    # ── 3. Process in chunks ──────────────────────────────────────────────────
    reader = pd.read_csv(
        DS2_ORIGINAL,
        sep=";",
        na_values=["?"],
        chunksize=chunksize,
    )

    for chunk_num, chunk in enumerate(reader):
        # Track which rows were originally missing (before forward-fill)
        chunk["_is_missing"] = chunk["Global_active_power"].isna()

        # Forward-fill missing within the chunk
        chunk.ffill(inplace=True)
        chunk.bfill(inplace=True)  # fallback for leading NaNs

        # Rename columns to standard key field names
        chunk.rename(columns=COLUMN_RENAME, inplace=True)

        # ── Build timestamp ───────────────────────────────────────────────────
        chunk["timestamp"] = pd.to_datetime(
            chunk["date"] + " " + chunk["time"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )

        # ── demand_load: alias of power_consumption ───────────────────────────
        chunk["demand_load"] = chunk["power_consumption"]

        # ── grid_frequency: 50 Hz ± voltage-deviation-correlated noise ────────
        voltage_dev = (chunk["voltage"] - voltage_nominal) / voltage_nominal
        chunk["grid_frequency"] = (
            50.0 - voltage_dev * 0.5
            + rng.normal(0, 0.02, len(chunk))
        ).round(3)

        # ── region & equipment_type (single household source) ─────────────────
        chunk["region"]         = "Zone_A_Residential"
        chunk["equipment_type"] = "smart_meter"

        # ── transformer_status & outage_event ─────────────────────────────────
        chunk["transformer_status"] = _derive_transformer_status(chunk["voltage"])
        chunk["outage_event"]       = _derive_outage_event(chunk)

        # Drop internal helper column
        chunk.drop(columns=["_is_missing"], inplace=True)

        # Write chunk to CSV
        mode   = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(DS2_AUGMENTED, mode=mode, header=header, index=False)
        first_chunk = False

        total_rows += len(chunk)
        logger.info(f"Chunk {chunk_num + 1}: processed {len(chunk)} rows (total: {total_rows:,})")

    log_pipeline_event(
        REQUEST_ID, "DS2 Augmentation", "complete",
        {"total_rows": total_rows, "output": str(DS2_AUGMENTED)},
    )
    logger.info(f"DS2 augmentation complete. Total rows: {total_rows:,}")


if __name__ == "__main__":
    run_augmentation(force=False)
