"""
augment_ds1.py — Augment the Smart Grid Stability dataset (DS1).

Input : data/source_files/smart_grid_stability_augmented.csv  (14 cols, 60K rows)
Output: data/smart_grid_stability_augmented.csv               (20 cols, 60K rows)
Backup: data/originals/smart_grid_stability_augmented_ORIGINAL.csv

Augmented columns added (6):
  timestamp        — synthetic hourly timestamps 2020-01-01 → 2023-12-31
  region           — Zone_A/B/C/D (unstable rows weighted to B & C)
  equipment_type   — derived from node role features (tau1, g-values)
  transformer_status — derived from stabf + stab score
  grid_frequency   — 50 Hz base + stab-proportional deviation + Gaussian noise
  outage_event     — derived from transformer_status

Run:
    python -m src.data.augment_ds1
"""

import shutil
import numpy as np
import pandas as pd

from src.config import DS1_AUGMENTED, DS1_ORIGINAL, ORIGINALS_DIR
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)
REQUEST_ID = "augment_ds1"


def _assign_region(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Assign Zone_A/B/C/D to each row.
    Unstable rows are weighted toward Zone_B and Zone_C to reflect
    the realistic pattern that instability clusters geographically.
    """
    zones = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
    regions = pd.Series(index=df.index, dtype=str)

    # Stable rows: uniform distribution
    stable_mask = df["stabf"] == "stable"
    stable_idx = df.index[stable_mask]
    regions[stable_idx] = rng.choice(zones, size=stable_mask.sum())

    # Unstable rows: 35% B, 35% C, 15% A, 15% D
    unstable_idx = df.index[~stable_mask]
    regions[unstable_idx] = rng.choice(
        zones, size=(~stable_mask).sum(), p=[0.15, 0.35, 0.35, 0.15]
    )
    return regions


def _derive_equipment_type(df: pd.DataFrame) -> pd.Series:
    """
    Map node roles to equipment types using tau and g feature thresholds.
      - High tau1 (producer reaction time)  → transformer
      - High g-values (consumer elasticity) → smart_meter_bank
      - Low stab margin                     → substation (stressed node)
      - Default                             → distribution_unit
    """
    tau1_75 = df["tau1"].quantile(0.75)
    g_avg = (df["g2"] + df["g3"] + df["g4"]) / 3
    g_75 = g_avg.quantile(0.75)

    conditions = [
        df["tau1"] >= tau1_75,
        g_avg >= g_75,
        df["stab"].abs() < 0.005,  # near-boundary nodes → substation
    ]
    choices = ["transformer", "smart_meter_bank", "substation"]
    return pd.Series(
        np.select(conditions, choices, default="distribution_unit"),
        index=df.index,
    )


def _derive_transformer_status(df: pd.DataFrame) -> pd.Series:
    """
    Map stability label + stab score to transformer operational status.

    stable   + stab < 0.02                       → normal
    stable   + stab >= 0.02                       → monitoring
    unstable + stab in (-0.01, 0.04)              → degraded
    unstable + stab >= 0.04                       → overloaded
    unstable + stab >= 0.07                       → critical
    """
    status = pd.Series("normal", index=df.index)
    s = df["stab"]
    sf = df["stabf"]

    status[sf == "stable"]                            = "normal"
    status[(sf == "stable") & (s >= 0.02)]            = "monitoring"
    status[(sf == "unstable")]                        = "degraded"
    status[(sf == "unstable") & (s >= 0.04)]          = "overloaded"
    status[(sf == "unstable") & (s >= 0.07)]          = "critical"
    return status


def _derive_outage_event(transformer_status: pd.Series) -> pd.Series:
    """
    Map transformer_status to outage event classification.
    """
    mapping = {
        "normal":     "no_event",
        "monitoring": "voltage_deviation",
        "degraded":   "voltage_deviation",
        "overloaded": "partial_outage",
        "critical":   "full_outage",
    }
    return transformer_status.map(mapping)


def run_augmentation(force: bool = False) -> pd.DataFrame:
    """
    Run the DS1 augmentation pipeline.

    Args:
        force: If True, re-augment even if the output file already exists.

    Returns:
        The augmented DataFrame (20 columns, 60K rows).
    """
    if DS1_AUGMENTED.exists() and not force:
        logger.info("DS1 augmented file already exists — loading from disk.")
        return pd.read_csv(DS1_AUGMENTED)

    logger.info("Starting DS1 augmentation pipeline.")
    log_pipeline_event(
        REQUEST_ID, "DS1 Augmentation", "start",
        {"source": str(DS1_ORIGINAL), "output": str(DS1_AUGMENTED)},
    )

    # ── 1. Load original ──────────────────────────────────────────────────────
    df = pd.read_csv(DS1_ORIGINAL)
    logger.info(f"Loaded DS1: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 2. Back up original ───────────────────────────────────────────────────
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = ORIGINALS_DIR / "smart_grid_stability_augmented_ORIGINAL.csv"
    if not backup_path.exists():
        shutil.copy(DS1_ORIGINAL, backup_path)
        logger.info(f"Backup written to {backup_path}")

    # ── 3. Sort by stab so chronological assignment is physics-consistent ─────
    df = df.sort_values("stab").reset_index(drop=True)

    # ── 4. Synthetic timestamps (2020-01-01 → 2023-12-31, hourly) ────────────
    rng = np.random.default_rng(seed=42)  # reproducible seed
    start = pd.Timestamp("2020-01-01")
    end   = pd.Timestamp("2023-12-31 23:00:00")
    timestamps = pd.date_range(start, end, periods=len(df))
    df["timestamp"] = timestamps
    logger.info("Timestamps assigned (2020–2023).")

    # ── 5. Region ─────────────────────────────────────────────────────────────
    df["region"] = _assign_region(df, rng)
    logger.info(f"Region distribution:\n{df['region'].value_counts().to_dict()}")

    # ── 6. Equipment type ─────────────────────────────────────────────────────
    df["equipment_type"] = _derive_equipment_type(df)
    logger.info(f"Equipment types:\n{df['equipment_type'].value_counts().to_dict()}")

    # ── 7. Transformer status ─────────────────────────────────────────────────
    df["transformer_status"] = _derive_transformer_status(df)
    logger.info(f"Transformer status:\n{df['transformer_status'].value_counts().to_dict()}")

    # ── 8. Grid frequency (50 Hz base + stab deviation + noise) ──────────────
    # When stab is high (unstable), frequency deviates more from nominal.
    stab_norm = (df["stab"] - df["stab"].min()) / (df["stab"].max() - df["stab"].min())
    freq_deviation = (stab_norm - 0.5) * 2.4  # range: ±1.2 Hz
    noise = rng.normal(0, 0.05, len(df))
    df["grid_frequency"] = (50.0 + freq_deviation + noise).round(3)
    logger.info(f"Grid frequency range: {df['grid_frequency'].min():.2f}–{df['grid_frequency'].max():.2f} Hz")

    # ── 9. Outage event ───────────────────────────────────────────────────────
    df["outage_event"] = _derive_outage_event(df["transformer_status"])
    logger.info(f"Outage events:\n{df['outage_event'].value_counts().to_dict()}")

    # ── 10. Save augmented file ───────────────────────────────────────────────
    DS1_AUGMENTED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DS1_AUGMENTED, index=False)
    logger.info(f"Augmented DS1 saved: {DS1_AUGMENTED} ({df.shape[0]} rows × {df.shape[1]} cols)")

    log_pipeline_event(
        REQUEST_ID, "DS1 Augmentation", "complete",
        {"rows": len(df), "cols": len(df.columns), "output": str(DS1_AUGMENTED)},
    )
    return df


if __name__ == "__main__":
    run_augmentation(force=False)
