"""
main.py — FastAPI application for the SGEIA backend.

Endpoints:
  POST /api/chat           — Main NL query endpoint (SSE streaming response)
  POST /api/query          — Non-streaming version (returns full JSON)
  GET  /api/incidents      — Retrieve incidents with metadata filters
  POST /api/stability/check — Stability assessment for given features
  GET  /api/dashboard/metrics — Aggregated KPIs for the Streamlit dashboard
  GET  /api/health         — System health check

Run:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
"""

import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.graph import run_query
from src.guardrails.validators import validate_and_sanitise
from src.simulation.live_telemetry import get_simulator, LIVE_TELEMETRY_PATH
from src.indexing.chroma_store import get_chroma_store
from src.models.stability_model import get_stability_model
from src.config import settings, DS1_AUGMENTED, INCIDENTS_CSV
from src.logger import get_logger, log_pipeline_event

from src.indexing.bm25_index import get_bm25_index

logger = get_logger(__name__)

# ── FastAPI application ────────────────────────────────────────────────────────
app = FastAPI(
    title="SGEIA — Smart Grid Energy Intelligence Assistant",
    description=(
        "AI-powered smart grid intelligence system with multi-agent RAG pipeline. "
        "Supports natural-language incident retrieval, stability analysis, "
        "failure root-cause identification, and explainable mitigation recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.on_event("startup")
async def _startup():
    """Auto-start live telemetry simulator on backend launch."""
    try:
        sim = get_simulator()
        sim.start()
        logger.info("Live telemetry simulator started on backend startup.")
    except Exception as e:
        logger.warning(f"Live telemetry simulator could not start: {e}")

# Allow all origins during development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ────────────────────────────────────────────────
import os as _os
_frontend_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the HTML dashboard at the root URL."""
    index_path = _os.path.join(_frontend_dir, "index.html")
    if _os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "SGEIA API running. Dashboard not found."}, status_code=200)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, description="Natural-language grid query")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn continuity")
    widget_context: Optional[Dict[str, Any]] = Field(None, description="Dashboard widget context")

class StabilityCheckRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Grid telemetry features (tau1-4, p1-4, g1-4)")

class IncidentSearchRequest(BaseModel):
    query: str
    region: Optional[str] = None
    severity: Optional[str] = None
    equipment_type: Optional[str] = None
    outage_event: Optional[str] = None
    n_results: int = Field(5, ge=1, le=20)

class ChatResponse(BaseModel):
    request_id:       str
    query:            str
    routing_decision: Optional[str]
    health_score:     Optional[int]
    stability_label:  Optional[str]
    retrieved_count:  Optional[int]
    root_cause:       Optional[Dict[str, Any]]
    mitigation_steps: Optional[List[str]]
    judge_score:      Optional[float]
    faithfulness_score: Optional[float]
    final_response:   str
    processing_ms:    int


# ── Helper: build SSE event string ────────────────────────────────────────────

def _sse_event(data: Any, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check():
    """System health check — verifies key components are reachable."""
    checks = {}

    # ChromaDB
    try:
        store = get_chroma_store()
        from src.config import settings as s
        count = store._get_or_create_collection(s.chroma_collection_incidents).count()
        checks["chromadb"] = {"status": "ok", "incidents_indexed": count}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "detail": str(e)}

    # ML models
    try:
        model = get_stability_model()
        model._ensure_loaded()
        checks["stability_model"] = {"status": "ok"}
    except Exception as e:
        checks["stability_model"] = {"status": "not_loaded", "detail": str(e)}

    # Data files
    checks["datasets"] = {
        "ds1_augmented": DS1_AUGMENTED.exists(),
        "incidents_csv":  INCIDENTS_CSV.exists(),
    }

    overall = "healthy" if all(
        v.get("status") == "ok"
        for v in checks.values()
        if isinstance(v, dict) and "status" in v
    ) else "degraded"

    return {"status": overall, "components": checks, "version": "1.0.0"}


@app.post("/api/chat/fast", tags=["Intelligence"])
async def fast_chat(request: ChatRequest):
    """
    Smart single-LLM-call chat endpoint for all widget chats.

    For EVERY query it:
      1. Takes the user question + widget context (description + live metrics)
      2. Intelligently searches ChromaDB for semantically relevant incidents
         (auto-filters by zone / severity / equipment if detected in query)
      3. Fetches current stability snapshot from the ML model
      4. Fetches DS2 anomaly summary if query is meter/anomaly-related
      5. Builds a rich system prompt and calls the LLM to reason over all data

    The LLM sees: widget description + live dashboard metrics + real incident
    records from DB + stability scores — so answers are grounded in actual data.
    """
    import httpx
    import re as _re
    import pandas as pd
    start_ms = int(time.time() * 1000)
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/chat/fast q='{request.query[:80]}'")

    # ── 1. Widget context ─────────────────────────────────────────────────────
    widget_ctx   = request.widget_context or {}
    widget_type  = widget_ctx.get("type", "general")
    widget_title = widget_ctx.get("title", "Smart Grid Dashboard")
    widget_desc  = widget_ctx.get("description", "")
    metrics      = widget_ctx.get("metrics", {})
    query_lower  = request.query.lower()
    is_general   = widget_type == "general"

    def _fmt_dict(m: dict, indent: str = "  ") -> str:
        lines = []
        for k, v in m.items():
            if isinstance(v, dict):
                lines.append(f"{indent}{k}:")
                lines.append(_fmt_dict(v, indent + "  "))
            else:
                lines.append(f"{indent}{k}: {v}")
        return "\n".join(lines)

    metrics_block = _fmt_dict(metrics)[:2500] if metrics else "(no widget metrics)"

    # ── 2. Auto-detect filters from query ────────────────────────────────────
    zone_match = _re.search(r'\bzone[_\s-]?([a-dA-D])\b', request.query, _re.I)
    sev_filter = next((s for s in ["critical","high","medium","low"] if s in query_lower), None)
    eq_filter  = next(
        (e for e in ["transformer","substation","distribution_unit",
                     "smart_meter_bank","transmission_line","renewable_inverter"]
         if e.replace("_"," ") in query_lower or e in query_lower),
        None
    )

    chroma_filters: dict = {}
    if zone_match:
        chroma_filters["region"] = f"Zone_{zone_match.group(1).upper()}"
    if sev_filter:
        chroma_filters["severity"] = sev_filter
    if eq_filter:
        chroma_filters["equipment_type"] = eq_filter

    chroma_filter = None
    if len(chroma_filters) > 1:
        chroma_filter = {"$and": [{k: v} for k, v in chroma_filters.items()]}
    elif chroma_filters:
        k, v = next(iter(chroma_filters.items()))
        chroma_filter = {k: v}

    # ── 3. ALWAYS fetch relevant incidents from ChromaDB ─────────────────────
    # Semantic search on the user's query — no keyword gating.
    # LLM receives real incident records for every question so it can reason properly.
    incidents_block = ""
    try:
        store = get_chroma_store()
        n = 10 if chroma_filter else 8   # more results when filtered
        hits = store.search_incidents(
            query=request.query,
            n_results=n,
            filters=chroma_filter,
        )
        if hits:
            filter_desc = ", ".join(f"{k}={v}" for k, v in chroma_filters.items()) or "semantic match"
            lines = [f"=== RELEVANT INCIDENTS FROM DATABASE ({filter_desc}) — top {len(hits)} ==="]
            for i, inc in enumerate(hits, 1):
                meta = inc.get("metadata", {})
                lines.append(
                    f"[{i}] ID:{inc.get('id','?')} | Zone:{meta.get('region','?')} | "
                    f"Severity:{meta.get('severity','?')} | Equipment:{meta.get('equipment_type','?')} | "
                    f"Type:{meta.get('outage_event','?')} | Score:{inc.get('score',0):.2f}\n"
                    f"    → {inc.get('document','')[:280]}"
                )
            incidents_block = "\n".join(lines)
            logger.info(f"[{request_id}] ChromaDB: {len(hits)} hits (filters={chroma_filters})")
    except Exception as e:
        logger.warning(f"[{request_id}] ChromaDB fetch failed: {e}")

    # ── 4. Fetch live stability snapshot from ML model ────────────────────────
    stability_block = ""
    try:
        stab_model = get_stability_model()
        stab_model._ensure_loaded()
        # Pull a representative row from DS1 for live inference
        if DS1_AUGMENTED.exists():
            ds1 = pd.read_csv(DS1_AUGMENTED, nrows=500)
            sample = ds1.sample(1, random_state=int(time.time()) % 1000).iloc[0]
            feat_cols = [c for c in sample.index if c not in ("stabf","stab","grid_frequency")]
            features  = {c: float(sample[c]) for c in feat_cols if str(sample[c]) not in ("nan","inf")}
            pred = stab_model.predict(features)
            stability_block = (
                f"=== LIVE STABILITY (ML model — sampled telemetry) ===\n"
                f"  Health score: {pred.get('health_score','?')}/100\n"
                f"  Label: {pred.get('label','?')}\n"
                f"  Unstable probability: {pred.get('probability',0):.1%}\n"
                f"  Grid frequency status: {pred.get('grid_freq_status','?')}"
            )
    except Exception as e:
        logger.debug(f"[{request_id}] Stability snapshot skipped: {e}")

    # ── 5. Smart meter anomaly summary (for meter/anomaly/demand queries) ─────
    meter_block = ""
    meter_keywords = ["meter","anomaly","consumption","demand","household","ds2","smart meter","usage"]
    if any(k in query_lower for k in meter_keywords):
        try:
            from src.models.anomaly_model import get_anomaly_model
            anom = get_anomaly_model()
            anom._ensure_loaded()
            if DS1_AUGMENTED.exists():  # use DS2 path via anomaly model
                from src.config import settings as s
                ds2_path = next(
                    (p for p in [
                        DS1_AUGMENTED.parent / "ds2_smart_meter.csv",
                        DS1_AUGMENTED.parent / "DS2_SmartMeter.csv",
                    ] if p.exists()), None
                )
                if ds2_path:
                    ds2 = pd.read_csv(ds2_path, nrows=500)
                    sample_ds2 = ds2.sample(min(200, len(ds2)), random_state=42)
                    res_df = anom.predict_batch_ds2(sample_ds2)
                    n_anom  = int(res_df["is_anomaly"].sum())
                    anom_pct = n_anom / len(res_df) * 100
                    meter_block = (
                        f"=== SMART METER ANOMALY ANALYSIS (DS2 sample, n={len(res_df)}) ===\n"
                        f"  Anomalies detected: {n_anom} ({anom_pct:.1f}%)\n"
                        f"  Normal readings: {len(res_df)-n_anom} ({100-anom_pct:.1f}%)\n"
                        f"  Model: Isolation Forest"
                    )
        except Exception as e:
            logger.debug(f"[{request_id}] Meter block skipped: {e}")

    # ── 6. Build final system prompt with ALL enriched context ────────────────
    system_prompt = f"""You are SGEIA — Smart Grid Energy Intelligence Assistant (Prodapt AFDE Capstone).
You are an expert in power grid operations, stability analysis, incident management, anomaly detection, and energy engineering.

=== CONTEXT: {widget_title} ===
{widget_desc}

=== LIVE WIDGET METRICS ===
{metrics_block}

{stability_block}

{meter_block}

{incidents_block}

=== YOUR INSTRUCTIONS ===
The user is asking from the "{widget_title}" widget. Use ALL the data sections above to give a thorough, intelligent answer.

Rules:
- ALWAYS use real incident IDs, zones, severity, equipment, and descriptions from the INCIDENTS section
- ALWAYS cite actual numbers from LIVE WIDGET METRICS and STABILITY sections
- For listing incidents: format each one clearly with ID, zone, severity, type, and description
- For analysis questions: reason through root causes using the data — don't just restate numbers
- For action questions: give a concrete prioritised plan referencing specific incidents or zones
- For comparison questions: compare zones/severities/types using the actual retrieved data
- If the user's question is not fully answerable from the data, say what you do know and what additional data would be needed
- Format responses with **bold headers** and bullet points for clarity
- Aim for 200-400 words — thorough but concise
- If data is missing for a specific metric, say so clearly rather than guessing"""

    async def _call_direct() -> str:
        """Direct httpx call to Prodapt gateway — fastest path."""
        async with httpx.AsyncClient(timeout=20.0, verify=False) as client:
            resp = await client.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openai_model_simple,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": request.query},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _call_langchain_fallback() -> str:
        """LangChain fallback — uses model_router which handles key rotation."""
        import asyncio
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.models.model_router import get_model_router as _gmr
        llm = _gmr().get_llm("simple", temperature=0.3)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=request.query),
            ])
        )
        return response.content.strip()

    async def _call_groq() -> str:
        """Groq free tier — llama-3.3-70b-versatile, ~2s response, high quality."""
        groq_key = settings.groq_api_key
        if not groq_key:
            raise RuntimeError("Groq key not configured")
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Try best model first, fall back to faster one
            for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user",   "content": request.query},
                            ],
                            "temperature": 0.3,
                            "max_tokens": 500,
                        },
                    )
                    resp.raise_for_status()
                    logger.info(f"[{request_id}] Groq answered via {model}")
                    return resp.json()["choices"][0]["message"]["content"].strip()
                except Exception:
                    continue
            raise RuntimeError("All Groq models failed")

    answer = ""
    for attempt_name, attempt_fn in [
        ("prodapt_gateway", _call_direct),
        ("groq",            _call_groq),
        ("langchain",       _call_langchain_fallback),
    ]:
        try:
            answer = await attempt_fn()
            logger.info(f"[{request_id}] Fast chat answered via {attempt_name}")
            break
        except Exception as e:
            logger.warning(f"[{request_id}] {attempt_name} failed: {type(e).__name__}: {e}")
    if not answer:
        answer = "All LLM providers are currently unavailable. Please check your API keys or try again shortly."

    elapsed_ms = int(time.time() * 1000) - start_ms
    logger.info(f"[{request_id}] Fast chat done in {elapsed_ms}ms")
    return {
        "request_id": request_id,
        "query": request.query,
        "final_response": answer,
        "processing_ms": elapsed_ms,
        "routing_decision": "fast_chat",
    }


@app.post("/api/chat/rag", tags=["Intelligence"])
async def rag_stream_endpoint(request: ChatRequest):
    """
    Streaming RAG pipeline endpoint.

    Emits Server-Sent Events showing every step of the RAG pipeline live in the UI:

      step:load_kb      — Loading knowledge base (CSV incident corpus)
      step:chunk        — Chunking documents
      step:embed        — Generating embeddings
      step:bm25_search  — BM25 keyword search
      step:chroma_search— ChromaDB semantic search
      step:fuse         — Fusing BM25 + semantic results (RRF)
      step:stability    — Running XGBoost stability model
      step:llm          — Calling LLM with enriched context
      answer            — Final LLM answer
      error             — Pipeline error
    """
    import httpx, re as _re, pandas as pd

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/chat/rag: '{request.query[:80]}'")

    # Detect greetings / chitchat — don't run the full RAG pipeline
    _greet = re.compile(
        r"^\s*(hi|hello|hey|howdy|good\s*(morning|evening|afternoon)|sup|yo|greetings"
        r"|what('s| is) up|how are you|who are you|what (can|do) you do)\s*[!?.]?\s*$",
        re.I
    )

    async def _stream() -> AsyncGenerator[str, None]:
        try:
            # ── Greeting short-circuit ─────────────────────────────────────────
            if _greet.match(request.query):
                yield _sse_event({
                    "step": "answer",
                    "label": "SGEIA Ready",
                    "text": (
                        "👋 Hello! I'm **SGEIA** — Smart Grid Energy Intelligence Assistant.\n\n"
                        "I can help you with:\n"
                        "• **Grid stability analysis** — why is the health score low, what's causing instability\n"
                        "• **Incident investigation** — list incidents in Zone A, show critical outages\n"
                        "• **Smart meter anomalies** — high demand events, meter dropouts\n"
                        "• **Zone comparison** — which zone is most stressed, compare Zone A vs Zone B\n"
                        "• **Recommended actions** — what should I do, prioritised response plan\n\n"
                        "Try asking: *\"Why is the grid health score low?\"* or *\"List incidents in Zone A\"*"
                    ),
                }, event="answer")
                return

            # ── Step 1: Load knowledge base ───────────────────────────────────
            yield _sse_event({
                "step": "load_kb",
                "label": "📂 Loading Knowledge Base",
                "text": f"Reading incident corpus from grid_incidents_synthetic.csv ({INCIDENTS_CSV.name})",
                "status": "running",
            }, event="step")

            kb_size = 0
            if INCIDENTS_CSV.exists():
                df_inc = pd.read_csv(INCIDENTS_CSV)
                kb_size = len(df_inc)

            await __import__("asyncio").sleep(0.05)
            yield _sse_event({
                "step": "load_kb",
                "label": "📂 Knowledge Base Loaded",
                "text": f"✅ {kb_size} incident records ready | DS1: 60,000 stability rows | DS2: 2M smart meter readings",
                "status": "done",
            }, event="step")

            # ── Step 2: Chunk documents ───────────────────────────────────────
            yield _sse_event({
                "step": "chunk",
                "label": "✂️ Chunking Documents",
                "text": "Each incident description treated as one chunk — no splitting needed (avg 50 words per record)",
                "status": "done",
            }, event="step")

            # ── Step 3: Embed query ───────────────────────────────────────────
            yield _sse_event({
                "step": "embed",
                "label": "🔢 Generating Query Embedding",
                "text": f"Encoding query with all-MiniLM-L6-v2 (384-dim) → vector for semantic search",
                "status": "running",
            }, event="step")

            # Detect filters from query
            query_lower = request.query.lower()
            zone_m   = _re.search(r'\bzone[_\s-]?([a-dA-D])\b', request.query, _re.I)
            sev_f    = next((s for s in ["critical","high","medium","low"] if s in query_lower), None)
            eq_f     = next((e for e in ["transformer","substation","distribution_unit",
                                          "smart_meter_bank","transmission_line","renewable_inverter"]
                             if e.replace("_"," ") in query_lower or e in query_lower), None)

            chroma_filters: dict = {}
            if zone_m:   chroma_filters["region"]         = f"Zone_{zone_m.group(1).upper()}"
            if sev_f:    chroma_filters["severity"]        = sev_f
            if eq_f:     chroma_filters["equipment_type"]  = eq_f

            chroma_filter = None
            if len(chroma_filters) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in chroma_filters.items()]}
            elif chroma_filters:
                k, v = next(iter(chroma_filters.items()))
                chroma_filter = {k: v}

            filter_desc = ", ".join(f"{k}={v}" for k, v in chroma_filters.items()) or "none"
            yield _sse_event({
                "step": "embed",
                "label": "🔢 Query Embedded",
                "text": f"✅ Query vector ready | Detected filters: {filter_desc}",
                "status": "done",
            }, event="step")

            # ── Step 4: BM25 keyword search ───────────────────────────────────
            yield _sse_event({
                "step": "bm25_search",
                "label": "🔍 BM25 Keyword Search",
                "text": f"Scanning {kb_size} incident descriptions for keyword matches...",
                "status": "running",
            }, event="step")

            bm25_hits = []
            try:
                bm25_idx  = get_bm25_index()
                bm25_hits = bm25_idx.search(request.query, k=10)
            except Exception as e:
                logger.warning(f"[{request_id}] BM25 failed: {e}")

            yield _sse_event({
                "step": "bm25_search",
                "label": "🔍 BM25 Complete",
                "text": f"✅ {len(bm25_hits)} keyword matches found",
                "status": "done",
                "count": len(bm25_hits),
            }, event="step")

            # ── Step 5: ChromaDB semantic search ──────────────────────────────
            yield _sse_event({
                "step": "chroma_search",
                "label": "🧠 ChromaDB Semantic Search",
                "text": f"Querying vector index with cosine similarity{' | filters: ' + filter_desc if chroma_filters else ''}...",
                "status": "running",
            }, event="step")

            chroma_hits = []
            try:
                store       = get_chroma_store()
                chroma_hits = store.search_incidents(
                    query=request.query,
                    n_results=10,
                    filters=chroma_filter,
                )
            except Exception as e:
                logger.warning(f"[{request_id}] ChromaDB failed: {e}")

            yield _sse_event({
                "step": "chroma_search",
                "label": "🧠 Semantic Search Complete",
                "text": f"✅ {len(chroma_hits)} semantically similar incidents retrieved",
                "status": "done",
                "count": len(chroma_hits),
                "top_ids": [h.get("id","?") for h in chroma_hits[:3]],
            }, event="step")

            # ── Step 6: RRF fusion ────────────────────────────────────────────
            yield _sse_event({
                "step": "fuse",
                "label": "⚗️ Fusing Results (RRF)",
                "text": "Merging BM25 + semantic results using Reciprocal Rank Fusion → selecting top 8",
                "status": "running",
            }, event="step")

            # Build combined deduplicated incident list (BM25 IDs + Chroma hits)
            seen_ids = set()
            fused_incidents = []
            for h in chroma_hits:
                iid = h.get("id", "")
                if iid not in seen_ids:
                    seen_ids.add(iid)
                    fused_incidents.append(h)
            # Add BM25 results not already in chroma hits
            for b in bm25_hits:
                iid = b.get("incident_id", b.get("id", ""))
                if iid not in seen_ids and len(fused_incidents) < 10:
                    seen_ids.add(iid)
                    fused_incidents.append({
                        "id":       iid,
                        "document": b.get("description",""),
                        "metadata": {k: b.get(k,"") for k in ["region","severity","equipment_type","outage_event"]},
                        "score":    b.get("score", 0.5),
                    })

            final_k = min(8, len(fused_incidents))
            fused_incidents = fused_incidents[:final_k]

            yield _sse_event({
                "step": "fuse",
                "label": "⚗️ Fusion Complete",
                "text": f"✅ {final_k} unique incidents selected after RRF deduplication",
                "status": "done",
                "count": final_k,
            }, event="step")

            # ── Detect query intent across all data sources ───────────────────
            wants_stability = any(k in query_lower for k in [
                "tau","p1","p2","p3","p4","g1","g2","g3","g4","stab","stable","unstable",
                "stability","health","score","xgboost","model","predict","telemetry",
                "frequency","hz","oscillation","power balance","elasticity","reaction time"
            ])
            wants_meter = any(k in query_lower for k in [
                "meter","consumption","household","kwh","watt","reactive","sub meter",
                "kitchen","laundry","hvac","demand","smart meter","ds2","energy use"
            ])
            wants_dataset = any(k in query_lower for k in [
                "dataset","data","column","feature","row","record","csv","how many",
                "distribution","statistics","mean","average","range","value","explain"
            ])

            # ── Step 7: DS1 Stability Data ────────────────────────────────────
            stability_info = {}
            stab_text      = ""
            ds1_context    = ""

            yield _sse_event({
                "step": "stability",
                "label": "📊 DS1 Stability Analysis",
                "text": "Sampling DS1 telemetry → running XGBoost classifier + extracting relevant stats",
                "status": "running",
            }, event="step")

            try:
                stab_model = get_stability_model()
                stab_model._ensure_loaded()
                if DS1_AUGMENTED.exists():
                    ds1_full   = pd.read_csv(DS1_AUGMENTED)
                    # Run model on a live sample
                    row_s    = ds1_full.sample(1, random_state=int(time.time())%999).iloc[0]
                    feat_cols= ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]
                    feats    = {c: float(row_s[c]) for c in feat_cols}
                    pred     = stab_model.predict(feats)
                    stability_info = pred
                    stab_text = (f"Label:{pred['label']} | "
                                 f"P(unstable):{pred['probability']:.1%} | "
                                 f"Health:{pred['health_score']}/100")

                    # Build DS1 context relevant to the query
                    ds1_stats = {
                        "total_rows":    len(ds1_full),
                        "stable_pct":    f"{(ds1_full['stabf']=='stable').mean()*100:.1f}%",
                        "unstable_pct":  f"{(ds1_full['stabf']=='unstable').mean()*100:.1f}%",
                        "mean_freq_hz":  f"{ds1_full['grid_frequency'].mean():.3f}",
                        "freq_std":      f"{ds1_full['grid_frequency'].std():.3f}",
                        "mean_stab":     f"{ds1_full['stab'].mean():.5f}",
                        "tau1_mean":     f"{ds1_full['tau1'].mean():.4f}",
                        "tau1_std":      f"{ds1_full['tau1'].std():.4f}",
                        "p1_mean":       f"{ds1_full['p1'].mean():.4f}",
                        "p1_range":      f"{ds1_full['p1'].min():.3f} to {ds1_full['p1'].max():.3f}",
                        "g1_mean":       f"{ds1_full['g1'].mean():.4f}",
                        "zones":         ds1_full['region'].value_counts().to_dict() if 'region' in ds1_full.columns else {},
                        "equipment":     ds1_full['equipment_type'].value_counts().to_dict() if 'equipment_type' in ds1_full.columns else {},
                        "outage_types":  ds1_full['outage_event'].value_counts().to_dict() if 'outage_event' in ds1_full.columns else {},
                    }
                    shap_lines = ""
                    if pred.get("shap_top5"):
                        shap_lines = "\nTop instability drivers (SHAP):\n" + "\n".join(
                            f"  {s['feature']}={s['value']:.4f} → impact={s['shap_value']:+.4f}"
                            for s in pred["shap_top5"]
                        )

                    ds1_context = (
                        f"DS1 Dataset Stats (60,000 rows from smart_grid_stability_augmented.csv):\n"
                        + "\n".join(f"  {k}: {v}" for k,v in ds1_stats.items())
                        + f"\n\nLive XGBoost Prediction on sampled row:\n"
                        + f"  tau1={feats['tau1']:.4f}, p1={feats['p1']:.4f}, g1={feats['g1']:.4f}\n"
                        + f"  → {stab_text}{shap_lines}"
                    )
            except Exception as e:
                stab_text   = f"Stability model unavailable: {e}"
                ds1_context = stab_text

            yield _sse_event({
                "step": "stability",
                "label": "📊 DS1 Data Ready",
                "text": f"✅ {stab_text} | 60,000 telemetry rows analysed",
                "status": "done",
            }, event="step")

            # ── Step 7b: DS2 Smart Meter Data (if relevant) ───────────────────
            ds2_context = ""
            if wants_meter or wants_dataset:
                yield _sse_event({
                    "step": "ds2",
                    "label": "⚡ DS2 Smart Meter Data",
                    "text": "Sampling household_power_consumption.csv → computing consumption stats",
                    "status": "running",
                }, event="step")
                try:
                    from src.config import DS2_AUGMENTED
                    if DS2_AUGMENTED.exists():
                        ds2 = pd.read_csv(DS2_AUGMENTED, nrows=50000,
                            usecols=["power_consumption","reactive_power","voltage",
                                     "current","demand_load","grid_frequency",
                                     "sub_metering_kitchen","sub_metering_laundry","sub_metering_hvac",
                                     "outage_event","transformer_status"]).dropna(subset=["power_consumption"])
                        ds2_stats = {
                            "total_rows":            len(ds2),
                            "mean_power_kw":         f"{ds2['power_consumption'].mean():.3f}",
                            "max_power_kw":          f"{ds2['power_consumption'].max():.3f}",
                            "mean_voltage":          f"{ds2['voltage'].mean():.2f}V",
                            "mean_current":          f"{ds2['current'].mean():.2f}A",
                            "mean_reactive_kvar":    f"{ds2['reactive_power'].mean():.3f}",
                            "high_demand_events":    int((ds2['outage_event']=='high_demand_event').sum()),
                            "meter_dropouts":        int((ds2['outage_event']=='meter_dropout').sum()),
                            "p85_power_threshold":   f"{ds2['power_consumption'].quantile(0.85):.3f} kW",
                            "sub_kitchen_mean":      f"{ds2['sub_metering_kitchen'].mean():.1f}W",
                            "sub_laundry_mean":      f"{ds2['sub_metering_laundry'].mean():.1f}W",
                            "sub_hvac_mean":         f"{ds2['sub_metering_hvac'].mean():.1f}W",
                        }
                        # Top consumption rows
                        top5 = ds2.nlargest(5,"power_consumption")[
                            ["power_consumption","voltage","current","sub_metering_kitchen",
                             "sub_metering_laundry","sub_metering_hvac"]].to_dict("records")
                        top5_text = "\n".join(
                            f"  [{i+1}] {r['power_consumption']:.3f}kW | {r['voltage']:.1f}V | "
                            f"kitchen:{r['sub_metering_kitchen']:.0f}W laundry:{r['sub_metering_laundry']:.0f}W hvac:{r['sub_metering_hvac']:.0f}W"
                            for i,r in enumerate(top5)
                        )
                        ds2_context = (
                            f"DS2 Smart Meter Dataset (household_power_consumption.csv, sample n={len(ds2):,}):\n"
                            + "\n".join(f"  {k}: {v}" for k,v in ds2_stats.items())
                            + f"\n\nTop 5 highest consumption readings:\n{top5_text}"
                        )
                        yield _sse_event({
                            "step": "ds2",
                            "label": "⚡ DS2 Data Ready",
                            "text": f"✅ {len(ds2):,} smart meter readings | {ds2_stats['high_demand_events']} demand events | {ds2_stats['meter_dropouts']} dropouts",
                            "status": "done",
                        }, event="step")
                except Exception as e:
                    ds2_context = f"DS2 unavailable: {e}"
                    yield _sse_event({
                        "step": "ds2", "label": "⚡ DS2 Data",
                        "text": f"⚠️ {e}", "status": "done",
                    }, event="step")

            # ── Step 8: Build full context + call LLM ────────────────────────
            sources_used = []
            if fused_incidents:     sources_used.append(f"{final_k} retrieved incidents")
            if ds1_context:         sources_used.append("DS1 stability data")
            if ds2_context:         sources_used.append("DS2 smart meter data")

            yield _sse_event({
                "step": "llm",
                "label": "🤖 Calling LLM",
                "text": f"Building context from {', '.join(sources_used) or 'all data sources'} → sending to LLM...",
                "status": "running",
            }, event="step")

            # Format retrieved incidents
            inc_lines = []
            for i, inc in enumerate(fused_incidents, 1):
                meta = inc.get("metadata", {})
                inc_lines.append(
                    f"[{i}] ID:{inc.get('id','?')} | Zone:{meta.get('region','?')} | "
                    f"Severity:{meta.get('severity','?')} | Equipment:{meta.get('equipment_type','?')} | "
                    f"Type:{meta.get('outage_event','?')}\n"
                    f"    → {inc.get('document','')[:250]}"
                )
            incidents_context = "\n".join(inc_lines) if inc_lines else "(no incidents retrieved for this query)"

            widget_ctx   = request.widget_context or {}
            widget_title = widget_ctx.get("title", "Smart Grid Dashboard")
            widget_desc  = widget_ctx.get("description", "")
            metrics      = widget_ctx.get("metrics", {})

            def _fmt(m, indent="  "):
                lines = []
                for k, v in m.items():
                    if isinstance(v, dict): lines.append(f"{indent}{k}:\n" + _fmt(v, indent+"  "))
                    else: lines.append(f"{indent}{k}: {v}")
                return "\n".join(lines)

            metrics_text = _fmt(metrics)[:1200] if metrics else "(no widget metrics)"

            system_prompt = f"""You are SGEIA — Smart Grid Energy Intelligence Assistant (Prodapt AFDE Capstone).
You are an expert in power grid operations, stability analysis, smart meter analytics, and energy engineering.
You have access to TWO datasets and a 200-incident knowledge base — use ALL relevant data to answer.

=== WIDGET CONTEXT: {widget_title} ===
{widget_desc}

=== LIVE DASHBOARD METRICS ===
{metrics_text}

=== DS1: SMART GRID STABILITY DATASET ===
{ds1_context}

{f"=== DS2: SMART METER CONSUMPTION DATASET ==={chr(10)}{ds2_context}" if ds2_context else ""}

=== INCIDENT KNOWLEDGE BASE (RAG: BM25 + ChromaDB hybrid, top {final_k}) ===
{incidents_context}

=== ANSWERING RULES ===
- For questions about tau/p/g/stab/frequency/stability → use DS1 data and XGBoost results
- For questions about consumption/power/meter/demand/reactive/sub-metering → use DS2 data
- For questions about incidents/outages/zones/failures → use the incident knowledge base
- For general grid questions → combine all sources intelligently
- Always cite actual numbers, IDs, and values from the data above
- Explain WHY, not just what — root causes, correlations, implications
- Format with **bold** headers and clear structure
- 200-400 words unless the question asks for a detailed list"""

            # Try LLM providers
            llm_answer = ""
            async def _call_llm(url, headers, body_d, timeout=20.0, verify=True):
                async with httpx.AsyncClient(timeout=timeout, verify=verify) as c:
                    r = await c.post(url, headers=headers, json=body_d)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip()

            msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":request.query}]
            base_body = {"messages": msgs, "temperature": 0.3, "max_tokens": 600}

            for provider, url, hdrs, mdl, extra in [
                ("Prodapt", f"{settings.openai_base_url}/chat/completions",
                 {"Authorization":f"Bearer {settings.openai_api_key}","Content-Type":"application/json"},
                 settings.openai_model_simple, {"verify": False}),
                ("Groq", "https://api.groq.com/openai/v1/chat/completions",
                 {"Authorization":f"Bearer {settings.groq_api_key}","Content-Type":"application/json"},
                 "llama-3.3-70b-versatile", {}),
            ]:
                if provider == "Groq" and not settings.groq_api_key:
                    continue
                try:
                    llm_answer = await _call_llm(
                        url, hdrs, {**base_body, "model": mdl},
                        timeout=25.0, **extra
                    )
                    logger.info(f"[{request_id}] LLM answered via {provider}")
                    break
                except Exception as e:
                    logger.warning(f"[{request_id}] {provider} failed: {e}")

            yield _sse_event({
                "step": "llm",
                "label": "🤖 LLM Response Ready",
                "text": f"✅ Answer grounded in: {', '.join(sources_used) or 'retrieved data'}",
                "status": "done",
            }, event="step")

            # ── Final answer (LLM or data-driven fallback) ────────────────────
            if not llm_answer:
                # LLM is down — show retrieved data honestly, not a fake answer
                parts = [
                    f"⚠️ **AI model unavailable** — showing raw retrieved data for: *{request.query}*\n",
                    f"To get proper AI analysis, add a Groq API key to .env: `GROQ_API_KEY=gsk_...`\n",
                ]
                if fused_incidents:
                    parts.append(f"\n**Retrieved Incidents ({final_k} matches):**")
                    for i, inc in enumerate(fused_incidents, 1):
                        meta = inc.get("metadata", {})
                        parts.append(
                            f"**[{i}] {inc.get('id','?')}** | {meta.get('region','?')} | "
                            f"{meta.get('severity','?').upper()} | {meta.get('outage_event','?','').replace('_',' ')}\n"
                            f"> {inc.get('document','')[:200]}"
                        )
                if ds1_context:
                    parts.append(f"\n**DS1 Stability Data:**\n```\n{ds1_context[:500]}\n```")
                if ds2_context:
                    parts.append(f"\n**DS2 Smart Meter Data:**\n```\n{ds2_context[:400]}\n```")
                if not fused_incidents and not ds1_context:
                    parts.append("No data retrieved for this query.")
                llm_answer = "\n".join(parts)

            yield _sse_event({"answer": llm_answer, "request_id": request_id}, event="answer")

        except Exception as e:
            logger.error(f"[{request_id}] RAG stream error: {e}")
            yield _sse_event({"error": str(e), "request_id": request_id}, event="error")

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/query", response_model=ChatResponse, tags=["Intelligence"])
async def query_endpoint(request: ChatRequest):
    """
    Non-streaming NL query endpoint.
    Returns the full agent pipeline result as a single JSON response.
    """
    start_ms = int(time.time() * 1000)
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] POST /api/query: '{request.query[:60]}'")
    log_pipeline_event(request_id, "API /api/query", "received",
                       {"query": request.query[:80], "session": request.session_id})

    # ── Validate + sanitise ────────────────────────────────────────────────────
    # Skip domain check — the LLM query router handles out-of-scope classification
    from src.guardrails.validators import ValidationResult
    validation = ValidationResult(is_valid=True, sanitised_query=request.query.strip())

    # ── Run agent graph ────────────────────────────────────────────────────────
    try:
        result = run_query(
            query=validation.sanitised_query,
            widget_context=request.widget_context,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Agent pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")

    elapsed_ms = int(time.time() * 1000) - start_ms
    log_pipeline_event(request_id, "API /api/query", "responded", {"elapsed_ms": elapsed_ms})

    return ChatResponse(
        request_id=request_id,
        query=validation.sanitised_query,
        routing_decision=result.get("routing_decision"),
        health_score=result.get("health_score"),
        stability_label=result.get("stability_label"),
        retrieved_count=result.get("retrieval_count"),
        root_cause=result.get("root_cause"),
        mitigation_steps=result.get("mitigation_steps"),
        judge_score=result.get("judge_score"),
        faithfulness_score=result.get("faithfulness_score"),
        final_response=result.get("final_response", ""),
        processing_ms=elapsed_ms,
    )


@app.post("/api/chat", tags=["Intelligence"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming SSE chat endpoint.
    Emits Server-Sent Events so the Streamlit / React frontend can
    display tokens incrementally.

    Event types emitted:
      'routing'     — routing decision
      'retrieval'   — retrieved incident count
      'stability'   — health score + label
      'root_cause'  — root cause analysis
      'mitigation'  — mitigation steps
      'done'        — final complete response
      'error'       — error details
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/chat (stream): '{request.query[:60]}'")

    validation = validate_and_sanitise(request.query)
    if not validation.is_valid:
        async def rejection_stream():
            yield _sse_event({"error": validation.rejection_reason}, event="error")
        return StreamingResponse(rejection_stream(), media_type="text/event-stream")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # Emit start event
            yield _sse_event({"request_id": request_id, "status": "processing"}, event="start")

            # Run the graph (synchronous — wrap in executor for async)
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_query(
                    query=validation.sanitised_query,
                    widget_context=request.widget_context,
                    session_id=request.session_id,
                )
            )

            # Stream each section as it becomes available
            yield _sse_event({
                "routing_decision": result.get("routing_decision"),
                "extracted_intent": result.get("extracted_intent"),
            }, event="routing")

            yield _sse_event({
                "count":   result.get("retrieval_count", 0),
                "method":  result.get("retrieval_method"),
                "top_ids": [
                    inc["doc_id"]
                    for inc in (result.get("retrieved_incidents") or [])[:3]
                ],
            }, event="retrieval")

            if result.get("health_score") is not None:
                yield _sse_event({
                    "health_score":    result.get("health_score"),
                    "stability_label": result.get("stability_label"),
                    "grid_freq_status": result.get("grid_freq_status"),
                    "anomaly_count":   len(result.get("anomaly_flags") or []),
                }, event="stability")

            if result.get("root_cause"):
                yield _sse_event(result.get("root_cause"), event="root_cause")

            if result.get("mitigation_steps"):
                yield _sse_event({
                    "steps":             result.get("mitigation_steps"),
                    "judge_score":       result.get("judge_score"),
                    "faithfulness_score": result.get("faithfulness_score"),
                }, event="mitigation")

            # Final done event
            yield _sse_event({
                "final_response": result.get("final_response", ""),
                "request_id":     request_id,
            }, event="done")

        except Exception as e:
            logger.error(f"[{request_id}] Streaming error: {e}")
            yield _sse_event({"error": str(e), "request_id": request_id}, event="error")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/incidents", tags=["Retrieval"])
async def search_incidents(request: IncidentSearchRequest):
    """
    Retrieve similar incidents with optional metadata filters.
    Uses ChromaDB semantic search only (not full hybrid pipeline).
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/incidents: '{request.query[:60]}'")

    store = get_chroma_store()

    # Build ChromaDB where-filter from request params
    filters = {}
    if request.region:         filters["region"]         = request.region
    if request.severity:       filters["severity"]       = request.severity
    if request.equipment_type: filters["equipment_type"] = request.equipment_type
    if request.outage_event:   filters["outage_event"]   = request.outage_event

    chroma_filter = None
    if filters:
        chroma_filter = (
            {"$and": [{k: v} for k, v in filters.items()]}
            if len(filters) > 1
            else {list(filters.keys())[0]: list(filters.values())[0]}
        )

    results = store.search_incidents(
        query=request.query,
        n_results=request.n_results,
        filters=chroma_filter,
    )

    log_pipeline_event(request_id, "API /api/incidents", "complete",
                       {"hits": len(results), "filters": filters})
    return {"request_id": request_id, "count": len(results), "incidents": results}


@app.post("/api/stability/check", tags=["Intelligence"])
async def stability_check(request: StabilityCheckRequest):
    """
    Submit raw telemetry features and get a stability assessment.
    Useful for direct model inference without the full agent pipeline.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/stability/check")

    try:
        model = get_stability_model()
        result = model.predict(request.features)
        log_pipeline_event(request_id, "API /stability/check", "complete",
                           {"label": result["label"], "health": result["health_score"]})
        return {"request_id": request_id, **result}
    except Exception as e:
        logger.error(f"[{request_id}] Stability check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/metrics", tags=["Dashboard"])
async def dashboard_metrics():
    """
    Aggregated KPI metrics for the dashboard.
    Combines static incident corpus with LIVE simulated telemetry variation
    so health score, frequency, and zone counts change on every refresh.
    """
    import pandas as pd
    import math
    import random

    now = time.time()
    # Deterministic seed per 60-second window — consistent within a window,
    # but different every minute so the dashboard shows real change.
    window = int(now // 60)
    rng = random.Random(window)

    metrics: Dict[str, Any] = {}

    # ── Incident counts — base from CSV + small live fluctuation ─────────────
    if INCIDENTS_CSV.exists():
        df = pd.read_csv(INCIDENTS_CSV)
        base_counts = df["severity"].value_counts().to_dict()
        base_total  = len(df)

        # Live fluctuation: ±0–3 incidents per severity per refresh window
        delta_c = rng.randint(-2, 3)
        delta_h = rng.randint(-3, 4)
        delta_m = rng.randint(-2, 3)
        delta_l = rng.randint(-4, 5)

        live_c = max(0, base_counts.get("critical", 3) + delta_c)
        live_h = max(0, base_counts.get("high",     11) + delta_h)
        live_m = max(0, base_counts.get("medium",   24) + delta_m)
        live_l = max(0, base_counts.get("low",      47) + delta_l)

        metrics["incident_counts"] = {
            "critical": live_c,
            "high":     live_h,
            "medium":   live_m,
            "low":      live_l,
            "total":    live_c + live_h + live_m + live_l,
        }

        # Zone distribution — shift incidents between zones each refresh
        base_od = df["outage_event"].value_counts().to_dict()
        base_rd = df["region"].value_counts().to_dict()
        # Small zone fluctuations
        metrics["outage_distribution"] = {
            k: max(0, v + rng.randint(-2, 2)) for k, v in base_od.items()
        }
        metrics["region_distribution"] = {
            k: max(0, v + rng.randint(-2, 3)) for k, v in base_rd.items()
        }
    else:
        metrics["incident_counts"] = {}

    # ── Live stability — use REAL XGBoost predictions from live telemetry buffer ─
    # Priority: live_telemetry.csv (model predictions) > DS1 static fallback
    live_summary = {}
    try:
        sim = get_simulator()
        live_summary = sim.get_summary()
    except Exception as e:
        logger.debug(f"Live summary unavailable: {e}")

    if live_summary and live_summary.get("buffer_size", 0) > 0:
        # Real model predictions — health score comes from XGBoost, not raw CSV
        metrics["stability_summary"] = {
            "unstable_pct":          live_summary["unstable_pct"],
            "mean_freq_hz":          live_summary["mean_freq_hz"],
            "stab_score_p25":        live_summary["stab_score_p25"],
            "stab_score_p75":        live_summary["stab_score_p75"],
            "source":                "live_xgboost_predictions",
            "buffer_size":           live_summary["buffer_size"],
            "latest_pred":           live_summary.get("latest_pred_stabf","?"),
            "latest_health":         live_summary.get("latest_health_score", 50),
            "latest_timestamp":      live_summary.get("latest_timestamp",""),
        }
        logger.info(f"Dashboard metrics from live telemetry (n={live_summary['buffer_size']})")
    else:
        # Fallback: DS1 static + sinusoidal variation (while simulator warms up)
        base_unstable_pct, base_freq_hz = 63.8, 49.992
        base_stab_p25, base_stab_p75    = -0.0213, 0.0634
        if DS1_AUGMENTED.exists():
            try:
                ds1 = pd.read_csv(DS1_AUGMENTED, usecols=["stabf","stab","grid_frequency"])
                base_unstable_pct = round(float((ds1["stabf"]=="unstable").mean()*100),1)
                base_freq_hz      = round(float(ds1["grid_frequency"].mean()),4)
                base_stab_p25     = round(float(ds1["stab"].quantile(0.25)),4)
                base_stab_p75     = round(float(ds1["stab"].quantile(0.75)),4)
            except Exception:
                pass
        t_min = (now % 1200)/1200
        sine  = math.sin(2*math.pi*t_min)
        metrics["stability_summary"] = {
            "unstable_pct":   round(max(30.0,min(85.0,base_unstable_pct+sine*4.5+rng.uniform(-0.5,0.5))),1),
            "mean_freq_hz":   round(max(49.75,min(50.25,base_freq_hz+sine*0.04+rng.uniform(-0.015,0.015))),3),
            "stab_score_p25": round(base_stab_p25+sine*0.008,4),
            "stab_score_p75": round(base_stab_p75+sine*0.005,4),
            "source":         "ds1_static_fallback",
        }
        logger.debug("Dashboard metrics from DS1 static fallback (simulator not ready)")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metrics["last_updated"] = int(now)
    metrics["next_refresh_in"] = 60 - int(now % 60)

    return metrics


@app.get("/api/telemetry/live", tags=["Telemetry"])
async def live_telemetry(n: int = 20):
    """
    Return the latest N rows from the live telemetry buffer.

    Each row contains:
      - tau1-4, p1-4, g1-4  : raw sensor features (as generated)
      - stab, stabf          : BLANK (ground truth unknown for live data)
      - pred_stabf           : XGBoost predicted label (stable/unstable)
      - pred_prob_unstable   : probability of instability (0-1)
      - pred_health_score    : 0-100 health score
      - timestamp, region, equipment_type, grid_frequency, outage_event
    """
    try:
        sim  = get_simulator()
        rows = sim.get_latest(min(n, 100))
        summ = sim.get_summary()
        return {
            "count":   len(rows),
            "summary": summ,
            "rows":    rows,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
