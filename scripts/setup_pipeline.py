"""
setup_pipeline.py — One-time SGEIA pipeline initialisation script.

Runs all data preparation, model training, and indexing steps in the
correct dependency order so the backend is fully ready to serve requests.

Steps:
  1. Validate environment (.env / API keys)
  2. Augment DS1 (grid stability dataset) with metadata columns
  3. Augment DS2 (smart meter dataset) with metadata columns
  4. Generate synthetic incident corpus (requires OPENAI_API_KEY)
  5. Train XGBoost stability model + anomaly detection models
  6. Index incidents into ChromaDB + BM25
  7. Index DS1 stability records into ChromaDB (optional, slow)
  8. Run smoke-test API calls to verify the backend is live

Usage:
    # Full setup (requires backend NOT running yet)
    python scripts/setup_pipeline.py

    # Skip model training (models already exist)
    python scripts/setup_pipeline.py --skip-training

    # Skip incident generation (CSV already exists)
    python scripts/setup_pipeline.py --skip-incidents

    # Force re-run all steps even if outputs exist
    python scripts/setup_pipeline.py --force

    # Skip slow DS1 ChromaDB indexing
    python scripts/setup_pipeline.py --skip-stability-index

Environment:
    Ensure .env file exists in project root before running.
    Minimum required: OPENAI_API_KEY (for incident generation step 4).
"""

import argparse
import sys
import time
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _banner(title: str) -> None:
    """Print a section banner to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _step(n: int, label: str) -> None:
    print(f"\n[Step {n}] {label} ...")


def _ok(msg: str = "Done") -> None:
    print(f"  [OK] {msg}")


def _skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ── Step implementations ──────────────────────────────────────────────────────

def step_validate_env() -> bool:
    """
    Validate that the .env file exists and required variables are set.
    Returns True if valid, False if critical variables missing.
    """
    from dotenv import load_dotenv
    import os

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        _fail(f".env file not found at {env_path}. Copy .env.example and fill in your keys.")
        return False

    load_dotenv(env_path)
    _ok(f".env loaded from {env_path}")

    required = ["OPENAI_API_KEY"]
    optional = ["GOOGLE_API_KEY", "GROQ_API_KEY", "LANGCHAIN_API_KEY"]

    missing_required = [k for k in required if not os.getenv(k)]
    missing_optional = [k for k in optional if not os.getenv(k)]

    if missing_required:
        _fail(f"Missing required env vars: {missing_required}")
        return False
    if missing_optional:
        print(f"  ⚠️  Optional env vars not set (fallback models may not work): {missing_optional}")

    _ok("Environment validated")
    return True


def step_augment_ds1(force: bool = False) -> bool:
    """Augment DS1 (grid stability) with metadata columns."""
    try:
        from src.data.augment_ds1 import run_augmentation
        run_augmentation(force=force)
        _ok("DS1 augmentation complete")
        return True
    except Exception as e:
        _fail(str(e))
        return False


def step_augment_ds2(force: bool = False) -> bool:
    """Augment DS2 (smart meter) with metadata columns (chunked processing)."""
    try:
        from src.data.augment_ds2 import run_augmentation
        run_augmentation(force=force)
        _ok("DS2 augmentation complete")
        return True
    except Exception as e:
        _fail(str(e))
        return False


def step_generate_incidents(force: bool = False) -> bool:
    """Generate synthetic incident corpus via GPT-4o-mini."""
    try:
        from src.data.generate_incidents import run_generation
        run_generation(force=force, use_llm=True)
        _ok("Incident corpus generated")
        return True
    except Exception as e:
        _fail(str(e))
        return False


def step_train_models(force: bool = False) -> bool:
    """Train XGBoost stability model + Isolation Forest anomaly models."""
    success = True
    try:
        from src.models.stability_model import get_stability_model
        model = get_stability_model()
        model.train(force=force)
        _ok("Stability model trained")
    except Exception as e:
        _fail(f"Stability model: {e}")
        success = False

    try:
        from src.models.anomaly_model import get_anomaly_model
        amodel = get_anomaly_model()
        amodel.train(force=force)
        _ok("Anomaly models trained")
    except Exception as e:
        _fail(f"Anomaly model: {e}")
        success = False

    return success


def step_index_incidents(force: bool = False) -> bool:
    """Index incident CSV into ChromaDB + build BM25 index."""
    success = True
    try:
        from src.indexing.chroma_store import get_chroma_store
        store = get_chroma_store()
        store.index_incidents(force=force)
        _ok("Incidents indexed into ChromaDB")
    except Exception as e:
        _fail(f"ChromaDB incidents: {e}")
        success = False

    try:
        from src.indexing.bm25_index import get_bm25_index
        idx = get_bm25_index()
        idx.build(force=force)
        _ok("BM25 index built")
    except Exception as e:
        _fail(f"BM25 index: {e}")
        success = False

    return success


def step_index_stability(force: bool = False) -> bool:
    """
    Index a sample of DS1 rows into ChromaDB for stability context retrieval.
    This step is slow (~5-10 min for 5000 rows) — can be skipped.
    """
    try:
        from src.indexing.chroma_store import get_chroma_store
        store = get_chroma_store()
        store.index_stability(max_rows=5000)
        _ok("DS1 stability records indexed into ChromaDB")
        return True
    except Exception as e:
        _fail(str(e))
        return False


def step_smoke_test() -> bool:
    """
    Run quick smoke tests against the backend API.
    Requires the backend to be running: uvicorn src.api.main:app
    """
    import requests

    base = "http://127.0.0.1:8000"
    tests = [
        ("GET",  f"{base}/api/health",            None),
        ("GET",  f"{base}/api/dashboard/metrics",  None),
    ]
    all_ok = True
    for method, url, payload in tests:
        try:
            if method == "GET":
                resp = requests.get(url, timeout=10)
            else:
                resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print(f"  ✅ {method} {url} → 200 OK")
            else:
                print(f"  ⚠️  {method} {url} → {resp.status_code}")
                all_ok = False
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  {url} unreachable — start the backend first.")
            all_ok = False
        except Exception as e:
            print(f"  ❌ {url}: {e}")
            all_ok = False
    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SGEIA One-Time Pipeline Setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--force",                action="store_true",
                        help="Force re-run all steps even if outputs exist")
    parser.add_argument("--skip-incidents",       action="store_true",
                        help="Skip incident generation (CSV already exists)")
    parser.add_argument("--skip-training",        action="store_true",
                        help="Skip model training (model files already exist)")
    parser.add_argument("--skip-stability-index", action="store_true",
                        help="Skip slow DS1 ChromaDB indexing")
    parser.add_argument("--smoke-only",           action="store_true",
                        help="Only run smoke tests (assumes everything is already set up)")
    args = parser.parse_args()

    _banner("SGEIA — Pipeline Setup")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Force mode   : {args.force}")

    if args.smoke_only:
        _step(1, "Smoke test — checking API health")
        step_smoke_test()
        return

    total_start = time.time()
    results: dict = {}

    # Step 1: Validate env
    _step(1, "Validate environment (.env)")
    results["env"] = step_validate_env()
    if not results["env"]:
        print("\n⛔ Setup aborted: fix environment issues first.")
        sys.exit(1)

    # Step 2: Augment DS1
    _step(2, "Augment DS1 (grid stability dataset)")
    results["ds1"] = step_augment_ds1(force=args.force)

    # Step 3: Augment DS2
    _step(3, "Augment DS2 (smart meter dataset) — may take several minutes")
    results["ds2"] = step_augment_ds2(force=args.force)

    # Step 4: Generate incidents
    if args.skip_incidents:
        _step(4, "Generate incident corpus")
        _skip("--skip-incidents flag set")
        results["incidents"] = True
    else:
        _step(4, "Generate synthetic incident corpus via GPT-4o-mini (~$0.50)")
        results["incidents"] = step_generate_incidents(force=args.force)

    # Step 5: Train models
    if args.skip_training:
        _step(5, "Train ML models")
        _skip("--skip-training flag set")
        results["models"] = True
    else:
        _step(5, "Train XGBoost + Isolation Forest models")
        results["models"] = step_train_models(force=args.force)

    # Step 6: Index incidents
    _step(6, "Index incidents → ChromaDB + BM25")
    results["index"] = step_index_incidents(force=args.force)

    # Step 7: Index stability (optional)
    if args.skip_stability_index:
        _step(7, "Index DS1 stability records into ChromaDB")
        _skip("--skip-stability-index flag set")
        results["stability_index"] = True
    else:
        _step(7, "Index DS1 stability records into ChromaDB (slow — ~5-10 min)")
        results["stability_index"] = step_index_stability(force=args.force)

    # Step 8: Smoke test
    _step(8, "Smoke test (requires backend running in separate terminal)")
    print("  ℹ️  To start backend: uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")
    results["smoke"] = step_smoke_test()

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - total_start, 1)
    _banner(f"Setup Complete — {elapsed}s")
    for step, ok in results.items():
        icon = "[OK]  " if ok else "[FAIL]"
        print(f"  {icon} {step}")

    all_ok = all(results.values())
    if all_ok:
        print("\nSGEIA is ready to serve requests!")
        print("   Start backend : uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")
        print("   Start frontend: streamlit run frontend/app.py")
    else:
        failed_steps = [k for k, v in results.items() if not v]
        print(f"\n[WARNING] Some steps failed: {failed_steps}. Review errors above before starting the backend.")
        sys.exit(1)


if __name__ == "__main__":
    main()
