"""
generate_ds2_sample.py — Create a representative 50K-row DS2 sample for Render.

On Render, the full 268MB household_power_consumption.csv cannot be committed
to git. This script generates a realistic synthetic sample with all 16 augmented
columns so every DS2 query and anomaly model still works correctly.

Run automatically by build.sh if DS2 is absent:
    python scripts/generate_ds2_sample.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import DS2_AUGMENTED

NROWS = 50_000
SEED  = 42


def _transformer_status(voltage: pd.Series) -> pd.Series:
    s = pd.Series("normal", index=voltage.index)
    s[voltage < 230]  = "degraded"
    s[(voltage >= 230) & (voltage < 237)] = "monitoring"
    s[voltage > 245]  = "monitoring"
    return s


def _outage_event(power: pd.Series, rng: np.random.Generator) -> pd.Series:
    ev = pd.Series("no_event", index=power.index)
    ev[power > 8.0] = "high_demand_event"
    # ~0.5% random meter dropouts
    dropout_idx = rng.choice(len(power), size=max(1, int(NROWS * 0.005)), replace=False)
    ev.iloc[dropout_idx] = "meter_dropout"
    return ev


def generate(nrows: int = NROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Build datetime index (minute-level, starting 2007-01-01)
    dates = pd.date_range("2007-01-01 00:00", periods=nrows, freq="min")

    # Core electrical columns (based on real UCI dataset statistics)
    voltage            = rng.normal(240.8, 3.2, nrows).clip(215, 255)
    power_consumption  = rng.exponential(1.1, nrows).clip(0, 12)
    reactive_power     = (power_consumption * rng.uniform(0.05, 0.35, nrows)).clip(0, 5)
    current            = (power_consumption / (voltage / 1000)).clip(0, 60)
    sub_kitchen        = rng.exponential(0.6, nrows).clip(0, 40)
    sub_laundry        = rng.exponential(1.0, nrows).clip(0, 40)
    sub_hvac           = rng.exponential(6.5, nrows).clip(0, 20)

    # Augmented columns
    demand_load       = power_consumption + rng.normal(0, 0.05, nrows)
    grid_frequency    = rng.normal(50.00, 0.04, nrows).clip(49.5, 50.5)
    regions           = rng.choice(
        ["North_Zone","South_Zone","East_Zone","West_Zone","Central_Zone"], nrows
    )
    equip_types       = rng.choice(
        ["residential_meter","industrial_meter","commercial_meter","substation_rtm"], nrows,
        p=[0.50, 0.20, 0.20, 0.10]
    )

    df = pd.DataFrame({
        "date":                  dates.strftime("%d/%m/%Y"),
        "time":                  dates.strftime("%H:%M:%S"),
        "power_consumption":     power_consumption.round(3),
        "reactive_power":        reactive_power.round(3),
        "voltage":               voltage.round(3),
        "current":               current.round(3),
        "sub_metering_kitchen":  sub_kitchen.round(3),
        "sub_metering_laundry":  sub_laundry.round(3),
        "sub_metering_hvac":     sub_hvac.round(3),
        "timestamp":             dates.astype(str),
        "demand_load":           demand_load.round(3),
        "grid_frequency":        grid_frequency.round(4),
        "region":                regions,
        "equipment_type":        equip_types,
        "transformer_status":    _transformer_status(pd.Series(voltage)),
        "outage_event":          _outage_event(pd.Series(power_consumption), rng),
    })

    return df


def main() -> None:
    if DS2_AUGMENTED.exists():
        print(f"[SKIP] DS2 already exists at {DS2_AUGMENTED} — skipping generation.")
        return

    DS2_AUGMENTED.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DS2] Generating {NROWS:,}-row sample → {DS2_AUGMENTED}")
    df = generate()
    df.to_csv(DS2_AUGMENTED, index=False)
    print(f"[DS2] Done — {len(df):,} rows, {DS2_AUGMENTED.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
