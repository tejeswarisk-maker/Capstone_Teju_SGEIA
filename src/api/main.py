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
from src.indexing.chroma_store import get_chroma_store
from src.models.stability_model import get_stability_model
from src.config import settings, DS1_AUGMENTED, INCIDENTS_CSV
from src.logger import get_logger, log_pipeline_event

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
    Fast single-LLM-call chat endpoint.
    Used by widget chats and general chat for quick responses.
    Skips the full multi-agent pipeline — returns in 5-15s instead of 60-120s.
    """
    import asyncio
    start_ms = int(time.time() * 1000)
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] POST /api/chat/fast: '{request.query[:60]}'")

    # Build context-aware system prompt
    widget_ctx = request.widget_context or {}
    widget_title = widget_ctx.get("title", "Smart Grid Dashboard")
    widget_desc  = widget_ctx.get("description", "")
    metrics_json = json.dumps(widget_ctx.get("metrics", {}), default=str)[:800]

    is_general = widget_ctx.get("type") == "general"
    system_prompt = f"""You are SGEIA — Smart Grid Energy Intelligence Assistant.
You are an expert in power grid operations, stability analysis, incident management, and energy systems.

{'=== FULL DASHBOARD SNAPSHOT (all 12 widgets) ===' if is_general else '=== WIDGET CONTEXT: ' + widget_title + ' ==='}
{widget_desc}

Dashboard data:
{metrics_json}

{'You have visibility across ALL widgets — grid health, active incidents, zone status, stability trend, frequency, voltage, demand, equipment, anomaly feed, agent activity, and recommendations. Answer cross-widget questions by comparing and correlating this data.' if is_general else 'Answer questions specifically about this widget using the data above.'}

Be specific, technical, and actionable. Reference actual numbers from the data.
Keep responses under 250 words unless more detail is requested."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = __import__('src.models.model_router', fromlist=['get_model_router']).get_model_router().get_llm("simple", temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query),
        ]
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[{request_id}] Fast chat error: {e}")
        answer = f"I encountered an error processing your request: {str(e)}"

    elapsed_ms = int(time.time() * 1000) - start_ms
    return {
        "request_id": request_id,
        "query": request.query,
        "final_response": answer,
        "processing_ms": elapsed_ms,
        "routing_decision": "fast_chat",
    }


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
    Aggregated KPI metrics for the Streamlit dashboard.
    Returns summary stats from the incidents corpus and latest stability state.
    """
    import pandas as pd

    metrics: Dict[str, Any] = {}

    # Incident counts by severity
    if INCIDENTS_CSV.exists():
        df = pd.read_csv(INCIDENTS_CSV)
        severity_counts = df["severity"].value_counts().to_dict()
        metrics["incident_counts"] = {
            "critical": severity_counts.get("critical", 0),
            "high":     severity_counts.get("high", 0),
            "medium":   severity_counts.get("medium", 0),
            "low":      severity_counts.get("low", 0),
            "total":    len(df),
        }
        metrics["outage_distribution"] = df["outage_event"].value_counts().to_dict()
        metrics["region_distribution"] = df["region"].value_counts().to_dict()
    else:
        metrics["incident_counts"] = {}

    # DS1 stability summary
    if DS1_AUGMENTED.exists():
        ds1 = pd.read_csv(DS1_AUGMENTED, usecols=["stabf", "stab", "grid_frequency"])
        metrics["stability_summary"] = {
            "unstable_pct":  round(float((ds1["stabf"] == "unstable").mean() * 100), 1),
            "mean_freq_hz":  round(float(ds1["grid_frequency"].mean()), 3),
            "stab_score_p25": round(float(ds1["stab"].quantile(0.25)), 4),
            "stab_score_p75": round(float(ds1["stab"].quantile(0.75)), 4),
        }

    return metrics
