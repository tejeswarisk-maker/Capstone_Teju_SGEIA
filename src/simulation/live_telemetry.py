"""
live_telemetry.py — Live Grid Telemetry Simulator

Simulates ALL columns from smart_grid_stability_augmented.csv with realistic
distributions and correlations, then uses the trained XGBoost model to predict
the stability label (stabf) and margin (stab) — which are LEFT BLANK at
generation time, exactly as real sensor data would arrive.

Column generation strategy
──────────────────────────
Generated (sensor input):
  tau1-4          — reaction time constants        (mean-reverting random walk)
  p1-4            — power balance per node         (mean-reverting random walk)
  g1-4            — price elasticity coefficients  (mean-reverting random walk)
  grid_frequency  — grid Hz reading                (mean-reverting walk near 50 Hz)
  timestamp       — UTC time of reading
  region          — Zone_A/B/C/D                   (weighted by DS1 distribution)
  equipment_type  — type of equipment              (weighted by DS1 distribution)

LEFT BLANK (ground truth unknown for live data):
  stab            — continuous stability margin
  stabf           — stable / unstable label

Model-predicted (XGBoost fills these in):
  pred_stabf           — predicted label
  pred_stab            — predicted stability margin
  pred_prob_unstable   — probability of instability (0-1)
  pred_health_score    — 0-100 health score

Derived from prediction (correlated, not random):
  transformer_status   — normal / degraded / overloaded / critical
  outage_event         — no_event / voltage_deviation / partial_outage / full_outage

DS1 full-dataset distributions (60 000 rows, computed offline):
  region:             Zone_C=31.85%, Zone_B=31.11%, Zone_A=18.72%, Zone_D=18.32%
  equipment_type:     distribution_unit=51.65%, transformer=25%, smart_meter_bank=18.75%, substation=4.6%
  transformer_status: normal=36.2%, degraded=34.93%, overloaded=21.86%, critical=7.01%
  outage_event:       no_event=36.2%, voltage_deviation=34.93%, partial_outage=21.86%, full_outage=7.01%
  stabf:              unstable=63.8%, stable=36.2%
  grid_frequency:     mean=50.017, std=0.469, min=48.75, max=51.27
  stab:               mean=0.0157, std=0.0369, min=-0.0808, max=0.1094
"""

import csv
import math
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.config import DS1_AUGMENTED
from src.logger import get_logger

logger = get_logger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
LIVE_TELEMETRY_PATH = DS1_AUGMENTED.parent / "live_telemetry.csv"
MAX_ROWS            = 500
INTERVAL_SECONDS    = 30

# ── DS1 distributions (computed from all 60 000 rows) ─────────────────────────
_NUMERIC_DIST = {
    #            mean      std     min       max
    "tau1":     (4.116,  2.840,  0.502,   9.987),
    "tau2":     (4.073,  2.809,  0.500,   9.999),
    "tau3":     (4.072,  2.809,  0.500,   9.999),
    "tau4":     (4.072,  2.809,  0.500,   9.999),
    "p1":       (3.733,  0.754,  1.687,   5.750),   # producer (> 0)
    "p2":       (-1.244, 0.436, -1.999,  -0.500),   # consumer (< 0)
    "p3":       (-1.244, 0.436, -1.999,  -0.500),
    "p4":       (-1.244, 0.436, -1.999,  -0.500),
    "g1":       (0.411,  0.254,  0.050,   0.997),
    "g2":       (0.414,  0.257,  0.050,   0.999),
    "g3":       (0.415,  0.257,  0.050,   0.999),
    "g4":       (0.415,  0.257,  0.050,   0.999),
    "grid_frequency": (50.017, 0.469, 48.750, 51.273),
}

_REGION_WEIGHTS      = {"Zone_C": 0.3185, "Zone_B": 0.3111, "Zone_A": 0.1872, "Zone_D": 0.1832}
_EQUIP_WEIGHTS       = {"distribution_unit": 0.5165, "transformer": 0.25,
                         "smart_meter_bank": 0.1875, "substation": 0.046}

# Ordered by stability probability threshold
# stable → normal + no_event
# mildly unstable → degraded + voltage_deviation
# moderately unstable → overloaded + partial_outage
# severely unstable → critical + full_outage
_TRANSFORMER_STATUS_BY_PROB = [
    (0.40, "normal"),
    (0.65, "degraded"),
    (0.85, "overloaded"),
    (1.01, "critical"),
]
_OUTAGE_EVENT_BY_PROB = [
    (0.40, "no_event"),
    (0.65, "voltage_deviation"),
    (0.85, "partial_outage"),
    (1.01, "full_outage"),
]

FEATURE_COLS = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

# Final CSV column order — matches augmented DS1 + prediction columns appended
CSV_COLUMNS = [
    # Physics sensor inputs
    "tau1","tau2","tau3","tau4",
    "p1","p2","p3","p4",
    "g1","g2","g3","g4",
    # Target columns — BLANK for live data
    "stab","stabf",
    # Context metadata
    "timestamp","region","equipment_type",
    "transformer_status","grid_frequency","outage_event",
    # XGBoost predictions (model fills these in)
    "pred_stabf","pred_stab","pred_prob_unstable","pred_health_score",
]


def _weighted_choice(weights: dict, rng: random.Random) -> str:
    """Pick a key from a {key: probability} dict."""
    keys   = list(weights.keys())
    probs  = list(weights.values())
    cumul  = 0.0
    r      = rng.random()
    for k, p in zip(keys, probs):
        cumul += p
        if r <= cumul:
            return k
    return keys[-1]


def _threshold_lookup(prob: float, thresholds: list) -> str:
    for threshold, value in thresholds:
        if prob < threshold:
            return value
    return thresholds[-1][1]


class LiveTelemetrySimulator:
    """
    Generates realistic live grid telemetry with ALL columns from the
    augmented DS1 dataset, using the trained XGBoost model to predict
    stability (stabf/stab left blank, pred_* columns filled by model).
    """

    def __init__(self) -> None:
        self._model        = None
        self._rng          = random.Random()
        self._lock         = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running      = False
        self._ds1_rows: List[Dict] = []   # actual DS1 rows for replay
        self._ds1_index    = 0
        self._ds1_loaded   = False

    # ── Model loading ───────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from src.models.stability_model import get_stability_model
            m = get_stability_model()
            m._ensure_loaded()
            self._model = m
            logger.info("LiveTelemetrySimulator: XGBoost model loaded.")
        except RuntimeError:
            logger.warning("LiveTelemetrySimulator: model not trained — training now...")
            from src.models.stability_model import StabilityModel
            m = StabilityModel()
            m.train()
            self._model = m

    # ── DS1 data loading and replay ─────────────────────────────────────────────

    def _load_ds1(self) -> None:
        """
        Load all DS1 rows into memory and shuffle for non-sequential replay.
        This ensures live telemetry is grounded in your ACTUAL dataset values,
        not synthetic distributions.
        """
        if self._ds1_loaded:
            return
        if not DS1_AUGMENTED.exists():
            logger.warning("DS1 not found — falling back to synthetic generation.")
            self._ds1_loaded = True
            return
        with open(DS1_AUGMENTED, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._ds1_rows = list(reader)
        # Shuffle so replay doesn't follow the original file order
        self._rng.shuffle(self._ds1_rows)
        self._ds1_index  = 0
        self._ds1_loaded = True
        logger.info(f"DS1 loaded for live replay: {len(self._ds1_rows):,} rows")

    def _next_ds1_row(self) -> Dict[str, float]:
        """
        Return the next DS1 row as a dict of floats (sensor features only).
        Wraps around when all rows have been played back once, then re-shuffles
        so the sequence is always different.
        """
        if not self._ds1_rows:
            return {}
        if self._ds1_index >= len(self._ds1_rows):
            # Re-shuffle and start over — infinite loop through real data
            self._rng.shuffle(self._ds1_rows)
            self._ds1_index = 0
            logger.info("DS1 replay: wrapped around — reshuffled for next cycle.")

        raw = self._ds1_rows[self._ds1_index]
        self._ds1_index += 1

        result = {}
        for col in FEATURE_COLS + ["grid_frequency"]:
            try:
                result[col] = round(float(str(raw.get(col, 0)).strip().strip("[]")), 6)
            except (ValueError, TypeError):
                result[col] = _NUMERIC_DIST.get(col, (0, 0, 0, 0))[0]
        return result

    # ── Core generation ─────────────────────────────────────────────────────────

    def generate_one(self) -> Dict:
        """
        Generate one live telemetry row by replaying a real DS1 row.

        Flow:
          1. Load DS1 into memory (once), shuffle for random ordering
          2. Read next DS1 row — tau1-4, p1-4, g1-4, grid_frequency from REAL data
          3. Leave stab / stabf BLANK  (as if arriving from a live sensor)
          4. XGBoost predicts: pred_stabf, pred_stab, pred_prob_unstable, pred_health_score
          5. Derive transformer_status and outage_event from prediction (correlated)
          6. Metadata: use DS1 row's region/equipment_type where available,
             otherwise sample from DS1 distribution weights
        """
        self._load_model()
        self._load_ds1()

        # ── Step 1: read next real DS1 row ────────────────────────────────────
        ds1_row = self._next_ds1_row()

        if not ds1_row:
            # DS1 unavailable — fallback to distribution-based generation
            ds1_row = {col: round(_NUMERIC_DIST[col][0] + self._rng.gauss(0, _NUMERIC_DIST[col][1] * 0.1), 6)
                       for col in FEATURE_COLS + ["grid_frequency"]}

        features = {col: ds1_row[col] for col in FEATURE_COLS}

        # ── Step 2: stabf / stab = BLANK ─────────────────────────────────────
        # The DS1 row has ground-truth stabf/stab but we deliberately hide them
        # to simulate real live sensor data (ground truth not known at read time).
        row: Dict = {col: ds1_row[col] for col in FEATURE_COLS}
        row["stab"]  = ""   # ← BLANK — ground truth hidden
        row["stabf"] = ""   # ← BLANK — XGBoost will predict this

        # ── Step 3: XGBoost prediction ────────────────────────────────────────
        prob_unstable = 0.5
        try:
            pred = self._model.predict(features)
            prob_unstable             = pred["probability"]
            row["pred_stabf"]         = pred["label"]
            row["pred_stab"]          = round(pred["stab_score"], 6)
            row["pred_prob_unstable"] = round(prob_unstable, 4)
            row["pred_health_score"]  = pred["health_score"]
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            row["pred_stabf"]         = "unknown"
            row["pred_stab"]          = ""
            row["pred_prob_unstable"] = 0.5
            row["pred_health_score"]  = 50

        # ── Step 4: derive correlated metadata from prediction ─────────────────
        row["transformer_status"] = _threshold_lookup(prob_unstable, _TRANSFORMER_STATUS_BY_PROB)
        row["outage_event"]       = _threshold_lookup(prob_unstable, _OUTAGE_EVENT_BY_PROB)

        # ── Step 5: grid_frequency from the real DS1 row ──────────────────────
        row["grid_frequency"] = ds1_row.get("grid_frequency",
                                            round(_NUMERIC_DIST["grid_frequency"][0], 3))

        # ── Step 6: metadata ──────────────────────────────────────────────────
        row["timestamp"]      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

        # Use region/equipment_type from actual DS1 row where available
        raw_ds1_row = self._ds1_rows[max(0, self._ds1_index - 1)] if self._ds1_rows else {}
        row["region"]         = raw_ds1_row.get("region") or _weighted_choice(_REGION_WEIGHTS, self._rng)
        row["equipment_type"] = raw_ds1_row.get("equipment_type") or _weighted_choice(_EQUIP_WEIGHTS, self._rng)

        return row

    # ── CSV buffer ───────────────────────────────────────────────────────────────

    def _append_to_buffer(self, row: Dict) -> None:
        """Append row to live_telemetry.csv and trim to MAX_ROWS."""
        with self._lock:
            existing: List[Dict] = []
            if LIVE_TELEMETRY_PATH.exists():
                with open(LIVE_TELEMETRY_PATH, newline="", encoding="utf-8") as f:
                    existing = list(csv.DictReader(f))

            existing.append(row)
            if len(existing) > MAX_ROWS:
                existing = existing[-MAX_ROWS:]

            LIVE_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LIVE_TELEMETRY_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing)

    # ── Background thread ────────────────────────────────────────────────────────

    def _run(self) -> None:
        logger.info(f"LiveTelemetrySimulator started — replaying DS1 rows, interval={INTERVAL_SECONDS}s, buffer={MAX_ROWS}")
        # Generate first reading immediately on start
        while self._running:
            try:
                row = self.generate_one()
                self._append_to_buffer(row)
                logger.info(
                    f"[telemetry] stabf=BLANK → pred={row.get('pred_stabf','?')} "
                    f"prob={row.get('pred_prob_unstable','?')} "
                    f"health={row.get('pred_health_score','?')} "
                    f"freq={row.get('grid_frequency','?')} "
                    f"zone={row.get('region','?')} "
                    f"outage={row.get('outage_event','?')}"
                )
            except Exception as e:
                logger.error(f"LiveTelemetrySimulator tick error: {e}")
            time.sleep(INTERVAL_SECONDS)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="live-telemetry")
        self._thread.start()
        logger.info("LiveTelemetrySimulator thread started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Query helpers ────────────────────────────────────────────────────────────

    def get_latest(self, n: int = 20) -> List[Dict]:
        with self._lock:
            if not LIVE_TELEMETRY_PATH.exists():
                return []
            with open(LIVE_TELEMETRY_PATH, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return rows[-n:]

    def get_summary(self) -> Dict:
        """
        Aggregate summary used by /api/dashboard/metrics.
        Health score = real XGBoost output on live telemetry, not DS1 CSV stats.
        """
        with self._lock:
            if not LIVE_TELEMETRY_PATH.exists():
                return {}
            with open(LIVE_TELEMETRY_PATH, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        if not rows:
            return {}

        probs, healths, freqs, stabs = [], [], [], []
        outage_counts: Dict[str, int] = {}
        zone_counts:   Dict[str, int] = {}
        status_counts: Dict[str, int] = {}

        for r in rows:
            try:
                p = float(r["pred_prob_unstable"])
                probs.append(p)
                healths.append(int(r["pred_health_score"]))
                freqs.append(float(r["grid_frequency"]))
                ps = r.get("pred_stab", "")
                if ps: stabs.append(float(ps))
            except (ValueError, KeyError):
                pass

            oe = r.get("outage_event", "")
            if oe: outage_counts[oe] = outage_counts.get(oe, 0) + 1

            rg = r.get("region", "")
            if rg: zone_counts[rg] = zone_counts.get(rg, 0) + 1

            ts = r.get("transformer_status", "")
            if ts: status_counts[ts] = status_counts.get(ts, 0) + 1

        if not probs:
            return {}

        n         = len(probs)
        latest    = rows[-1]
        probs_s   = sorted(probs)

        return {
            "source":                 f"DS1_replay_{len(self._ds1_rows)}_rows" if self._ds1_rows else "synthetic_fallback",
            "ds1_rows_loaded":        len(self._ds1_rows),
            "ds1_rows_played":        self._ds1_index,
            "buffer_size":            n,
            "unstable_pct":           round(sum(1 for p in probs if p >= 0.5) / n * 100, 1),
            "mean_health_score":      round(sum(healths) / n, 1),
            "mean_freq_hz":           round(sum(freqs) / n, 3),
            "stab_score_p25":         round(probs_s[n // 4], 4),
            "stab_score_p75":         round(probs_s[3 * n // 4], 4),
            "outage_distribution":    outage_counts,
            "zone_distribution":      zone_counts,
            "transformer_status_dist": status_counts,
            "latest_pred_stabf":      latest.get("pred_stabf", "?"),
            "latest_prob_unstable":   float(latest.get("pred_prob_unstable", 0.5)),
            "latest_health_score":    int(latest.get("pred_health_score", 50)),
            "latest_freq":            latest.get("grid_frequency", "?"),
            "latest_outage":          latest.get("outage_event", "?"),
            "latest_zone":            latest.get("region", "?"),
            "latest_transformer":     latest.get("transformer_status", "?"),
            "latest_timestamp":       latest.get("timestamp", ""),
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_simulator: Optional[LiveTelemetrySimulator] = None


def get_simulator() -> LiveTelemetrySimulator:
    global _simulator
    if _simulator is None:
        _simulator = LiveTelemetrySimulator()
    return _simulator
