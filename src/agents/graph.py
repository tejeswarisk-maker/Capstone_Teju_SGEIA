"""
graph.py — LangGraph StateGraph for the SGEIA multi-agent pipeline.

Graph topology:
  START
    └── route_query (LLM Query Router)
          ├── [incident]      → retrieve_incidents → analyse_failure → generate_recommendations → synthesise
          ├── [stability]     → assess_stability   → analyse_failure → generate_recommendations → synthesise
          ├── [smart_meter]   → analyse_smart_meter → generate_recommendations → synthesise
          ├── [cross_domain]  → retrieve_incidents + assess_stability + analyse_smart_meter
          │                   → analyse_failure → generate_recommendations → synthesise
          └── [out_of_scope]  → out_of_scope_response → END

A2A escalation: Recommendation Agent can trigger re-retrieval if quality gates fail
(handled inside recommendation_agent.py's retry loop).

Checkpoint: MemorySaver enables conversation continuity within a session.
"""

import uuid
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState, initial_state
from src.agents.query_router       import route_query
from src.agents.retrieval_agent    import retrieve_incidents
from src.agents.stability_agent    import assess_stability
from src.agents.smart_meter_agent  import analyse_smart_meter
from src.agents.failure_agent      import analyse_failure
from src.agents.recommendation_agent import generate_recommendations
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)


# ── Routing function — reads routing_decision from state ──────────────────────

def _decide_next_after_router(
    state: AgentState,
) -> Literal["retrieve_incidents", "assess_stability", "analyse_smart_meter",
             "retrieve_and_assess", "out_of_scope_node"]:
    """
    Conditional edge function called after LLM Query Router.
    Maps routing_decision to the next graph node.
    """
    decision = state.get("routing_decision", "cross_domain")
    mapping = {
        "incident":     "retrieve_incidents",
        "stability":    "assess_stability",
        "smart_meter":  "analyse_smart_meter",
        "cross_domain": "retrieve_and_assess",   # fan-out node
        "out_of_scope": "out_of_scope_node",
    }
    return mapping.get(decision, "retrieve_and_assess")


# ── Fan-out node: run retrieval + stability + smart meter in sequence ─────────
# LangGraph does not natively support parallel fan-out without custom reducers,
# so we run all three agents sequentially in a single composite node.

def retrieve_and_assess(state: AgentState) -> AgentState:
    """
    Composite node for cross-domain queries.
    Runs: Grid Retrieval + Grid Stability + Smart Meter sequentially.
    """
    request_id = state.get("request_id", "N/A")
    logger.info(f"[{request_id}] Cross-domain fan-out: running all three pipeline agents.")

    state = retrieve_incidents(state)
    state = assess_stability(state)
    state = analyse_smart_meter(state)
    return state


# ── After stability-only path: still need failure analysis ────────────────────

def stability_to_failure(state: AgentState) -> AgentState:
    """Pass-through: stability → failure analysis."""
    return analyse_failure(state)


# ── Out-of-scope response ──────────────────────────────────────────────────────

def out_of_scope_node(state: AgentState) -> AgentState:
    """Return a helpful rejection message for out-of-scope queries."""
    request_id = state.get("request_id", "N/A")
    logger.info(f"[{request_id}] Query classified as out-of-scope.")
    log_pipeline_event(request_id, "Graph", "out_of_scope", {"query": state.get("query", "")[:80]})
    return {
        **state,
        "final_response": (
            "⚠️ This query appears to be outside the scope of the Smart Grid Energy "
            "Intelligence Assistant. I can help with:\n"
            "• Grid stability and outage analysis\n"
            "• Historical incident retrieval and pattern matching\n"
            "• Smart meter anomaly detection and demand analysis\n"
            "• Equipment health monitoring and mitigation recommendations\n\n"
            "Please rephrase your question in the context of power grid operations."
        ),
    }


# ── Synthesis node — builds the final formatted response ─────────────────────

def synthesise_response(state: AgentState) -> AgentState:
    """
    Assembles the final user-facing response from all agent outputs.
    Uses simple string formatting (no extra LLM call — already token-optimised).
    """
    request_id = state.get("request_id", "N/A")
    log_pipeline_event(request_id, "Synthesiser", "start", {})

    parts: list[str] = []

    # ── Grid Health ────────────────────────────────────────────────────────────
    health = state.get("health_score")
    label  = state.get("stability_label")
    if health is not None:
        colour = "🟢" if health >= 70 else ("🟡" if health >= 40 else "🔴")
        parts.append(
            f"## {colour} Grid Health: {health}/100  ({(label or 'N/A').upper()})\n"
            f"Stability probability (unstable): {state.get('stability_prob', 0):.1%} | "
            f"Frequency status: {state.get('grid_freq_status', 'N/A')}"
        )

    # ── Anomalies ─────────────────────────────────────────────────────────────
    anomalies = state.get("anomaly_flags", [])
    if anomalies:
        parts.append(
            f"## ⚡ Anomalies Detected: {len(anomalies)}\n"
            + "\n".join(
                f"- {a.get('timestamp','?')} | {a.get('equipment_type','?')} | "
                f"Status: {a.get('transformer_status','?')} | Score: {a.get('anomaly_score',0):.3f}"
                for a in anomalies[:5]
            )
        )

    # ── Retrieved Incidents ───────────────────────────────────────────────────
    incidents = state.get("retrieved_incidents", [])
    if incidents:
        parts.append("## 📋 Similar Historical Incidents")
        for inc in incidents[:3]:
            meta = inc.get("metadata", {})
            sim  = f"{inc.get('rrf_score', 0):.2%}" if inc.get('rrf_score') else "N/A"
            parts.append(
                f"**{inc['doc_id']}** ({meta.get('severity','?').upper()}) | "
                f"{meta.get('region','?')} | {meta.get('equipment_type','?')} | "
                f"Similarity: {sim}\n> {inc['document'][:200]}..."
            )

    # ── Root Cause ─────────────────────────────────────────────────────────────
    rc = state.get("root_cause", {})
    if rc and rc.get("probable_cause"):
        parts.append(
            f"## 🔍 Root Cause Analysis\n"
            f"**Probable cause:** {rc.get('probable_cause')}\n\n"
            f"**Evidence:** {rc.get('evidence', 'N/A')}\n\n"
            f"**Confidence:** {rc.get('confidence', 0):.0%} | "
            f"Related incidents: {', '.join(rc.get('related_incident_ids', [])) or 'None'}"
        )

    # ── Mitigation ────────────────────────────────────────────────────────────
    steps = state.get("mitigation_steps", [])
    if steps:
        judge = state.get("judge_score", 0)
        faith = state.get("faithfulness_score", 0)
        parts.append(
            f"## 🛠️ Mitigation Recommendations  "
            f"(Judge score: {judge:.1f}/5 | Faithfulness: {faith:.0%})\n"
            + "\n".join(steps)
        )

    # ── Smart Meter ───────────────────────────────────────────────────────────
    sm = state.get("smart_meter_summary")
    if sm:
        parts.append(f"## 📡 Smart Meter Analysis\n{sm}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap = state.get("shap_top5", [])
    if shap:
        parts.append(
            "## 📊 Key Instability Drivers (SHAP)\n"
            + " | ".join(
                f"`{s['feature']}`={s['value']:.3f} (SHAP {s['shap_value']:+.3f})"
                for s in shap
            )
        )

    final_response = "\n\n".join(parts) if parts else "No analysis data available."
    log_pipeline_event(request_id, "Synthesiser", "complete",
                       {"response_length": len(final_response)})

    return {**state, "final_response": final_response}


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph application with MemorySaver checkpoint.
    """
    workflow = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    workflow.add_node("route_query",           route_query)
    workflow.add_node("retrieve_incidents",    retrieve_incidents)
    workflow.add_node("assess_stability",      assess_stability)
    workflow.add_node("analyse_smart_meter",   analyse_smart_meter)
    workflow.add_node("retrieve_and_assess",   retrieve_and_assess)
    workflow.add_node("analyse_failure",       analyse_failure)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("synthesise_response",   synthesise_response)
    workflow.add_node("out_of_scope_node",     out_of_scope_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.add_edge(START, "route_query")

    # ── Conditional routing after LLM Query Router ────────────────────────────
    workflow.add_conditional_edges(
        "route_query",
        _decide_next_after_router,
        {
            "retrieve_incidents":  "retrieve_incidents",
            "assess_stability":    "assess_stability",
            "analyse_smart_meter": "analyse_smart_meter",
            "retrieve_and_assess": "retrieve_and_assess",
            "out_of_scope_node":   "out_of_scope_node",
        },
    )

    # ── Incident path ─────────────────────────────────────────────────────────
    workflow.add_edge("retrieve_incidents", "analyse_failure")

    # ── Stability path ────────────────────────────────────────────────────────
    workflow.add_edge("assess_stability",   "analyse_failure")

    # ── Smart meter path ──────────────────────────────────────────────────────
    workflow.add_edge("analyse_smart_meter", "generate_recommendations")

    # ── Cross-domain fan-out path ─────────────────────────────────────────────
    workflow.add_edge("retrieve_and_assess", "analyse_failure")

    # ── Failure → Recommendation → Synthesis ──────────────────────────────────
    workflow.add_edge("analyse_failure",       "generate_recommendations")
    workflow.add_edge("generate_recommendations", "synthesise_response")

    # ── All paths → END ───────────────────────────────────────────────────────
    workflow.add_edge("synthesise_response", END)
    workflow.add_edge("out_of_scope_node",   END)

    # Compile with memory checkpointing (enables session continuity)
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully.")
    return app


# ── Public API ─────────────────────────────────────────────────────────────────

_graph_app = None


def get_graph():
    """Return the compiled LangGraph application (singleton)."""
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


def run_query(
    query: str,
    widget_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> AgentState:
    """
    Run a user query through the full agent pipeline.

    Args:
        query:          Natural-language user query.
        widget_context: Optional dashboard widget context dict.
        session_id:     Session ID for LangGraph checkpointing.
                        If None, a new session UUID is generated.

    Returns:
        Final AgentState with all agent outputs populated.
    """
    request_id = str(uuid.uuid4())[:8]
    session_id = session_id or str(uuid.uuid4())

    logger.info(f"[{request_id}] New query received (session={session_id}): '{query[:80]}'")
    log_pipeline_event(request_id, "Graph", "query_start",
                       {"query": query[:100], "session_id": session_id})

    state = initial_state(request_id, query, widget_context)
    app   = get_graph()

    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke(state, config=config)

    log_pipeline_event(request_id, "Graph", "query_complete",
                       {"response_length": len(result.get("final_response", ""))})
    return result
