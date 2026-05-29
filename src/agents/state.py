"""
state.py — Shared AgentState TypedDict for the LangGraph pipeline.

Every node in the LangGraph graph reads from and writes to this state.
Only summary-level data (not full DataFrames) is passed between nodes
to keep token usage and memory footprint minimal.

Fields are populated progressively as the graph executes:
  1. query, request_id, widget_context   (set at ingestion)
  2. routing_decision, metadata_filters  (set by LLM Query Router)
  3. retrieved_incidents, similarity_scores (set by Grid Retrieval Agent)
  4. stability_score, anomaly_flags, shap_top5 (set by Grid Stability Agent)
  5. smart_meter_summary                 (set by Smart Meter Agent)
  6. root_cause                          (set by Failure Analysis Agent)
  7. mitigation_steps, judge_score       (set by Recommendation Agent)
  8. final_response, fallback_model_used (set by Orchestrator / synthesis)
"""

from typing import Any, Dict, List, Optional, TypedDict


class RetrievedIncident(TypedDict):
    """Lightweight representation of a retrieved incident document."""
    doc_id:    str
    document:  str          # the incident description text
    metadata:  Dict[str, Any]
    rrf_score: float        # fused BM25 + semantic score
    rerank_score: Optional[float]  # Cross-Encoder score (added after reranking)


class SHAPFeature(TypedDict):
    """A single SHAP feature importance entry."""
    feature:    str
    value:      float
    shap_value: float


class AgentState(TypedDict, total=False):
    """
    Full shared state passed between all LangGraph agent nodes.

    Fields marked Optional are not set in every query flow.
    The 'total=False' means all fields are optional at construction time —
    individual agents add their keys as they run.
    """

    # ── Input fields (set at API gateway) ────────────────────────────────────
    request_id:      str            # UUID for end-to-end traceability
    query:           str            # original user natural-language query
    widget_context:  Optional[Dict[str, Any]]  # dashboard widget context (if any)

    # ── Routing (set by LLM Query Router) ────────────────────────────────────
    routing_decision: Optional[str]         # e.g. "stability|incident|cross_domain"
    metadata_filters: Optional[Dict[str, Any]]  # ChromaDB where-filter
    extracted_intent: Optional[str]         # one-line summary of query intent

    # ── Grid Retrieval Agent outputs ──────────────────────────────────────────
    retrieved_incidents:  Optional[List[RetrievedIncident]]
    retrieval_method:     Optional[str]  # "hybrid" | "semantic" | "bm25"
    retrieval_count:      Optional[int]

    # ── Grid Stability Agent outputs ──────────────────────────────────────────
    stability_label:   Optional[str]    # "stable" | "unstable"
    stability_prob:    Optional[float]  # probability of unstable
    stab_score:        Optional[float]  # continuous stability margin
    health_score:      Optional[int]    # 0–100
    anomaly_flags:     Optional[List[Dict[str, Any]]]  # list of detected anomalies
    shap_top5:         Optional[List[SHAPFeature]]
    grid_freq_status:  Optional[str]    # "normal" | "deviation" | "excursion"

    # ── Smart Meter Agent outputs ─────────────────────────────────────────────
    smart_meter_summary:   Optional[str]   # prose summary of meter analysis
    anomaly_events_count:  Optional[int]
    demand_forecast:       Optional[Dict[str, Any]]  # {"next_1h": float, "next_24h": float}

    # ── Failure Analysis Agent outputs ────────────────────────────────────────
    root_cause: Optional[Dict[str, Any]]
    # root_cause schema:
    # {
    #   "probable_cause": str,
    #   "evidence":       str,
    #   "confidence":     float (0–1),
    #   "related_incident_ids": [str],
    # }

    # ── Recommendation Agent outputs ──────────────────────────────────────────
    mitigation_steps:   Optional[List[str]]  # numbered mitigation actions
    judge_score:        Optional[float]      # LLM-as-judge score (0–5)
    faithfulness_score: Optional[float]      # DeepEval faithfulness (0–1)
    regeneration_count: Optional[int]        # how many times mitigation was regenerated

    # ── Final output (set by synthesis step) ─────────────────────────────────
    final_response:      Optional[str]   # full formatted response to return to user
    fallback_model_used: Optional[str]   # which fallback model was triggered (if any)
    error_message:       Optional[str]   # populated if any agent step fails


def initial_state(request_id: str, query: str, widget_context: Optional[Dict] = None) -> AgentState:
    """
    Create a fresh AgentState with only the input fields populated.
    All other fields default to None.
    """
    return AgentState(
        request_id=request_id,
        query=query,
        widget_context=widget_context,
        regeneration_count=0,
    )
