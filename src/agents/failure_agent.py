"""
failure_agent.py — Failure Analysis Agent node.

Synthesises retrieved incidents + stability telemetry + SHAP features
into a structured root-cause analysis using GPT-4o (with fallback).

Output schema:
  {
    "probable_cause":       str,
    "evidence":             str,
    "confidence":           float (0–1),
    "related_incident_ids": [str],
  }
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.models.model_router import get_model_router
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)

FAILURE_SYSTEM_PROMPT = """You are a senior power grid engineer performing root-cause analysis.

You are given:
1. A utility engineer's query describing a grid issue.
2. Retrieved historical incidents similar to the current issue.
3. Real-time telemetry summary (stability score, anomaly flags, frequency status).
4. SHAP feature importances explaining which telemetry variables most influenced
   the instability prediction.

Your task: Identify the most probable root cause of the described grid failure.

Respond in JSON with this exact structure:
{
  "probable_cause": "<one clear sentence describing the root cause>",
  "evidence": "<2–3 sentences citing specific incident IDs, telemetry values, and SHAP features as evidence>",
  "confidence": <float 0.0–1.0>,
  "related_incident_ids": ["INC-XXXX", ...]
}

Be technically precise. Reference actual values from the provided context."""


def _build_context(state: AgentState) -> str:
    """Build the context message for the Failure Analysis Agent."""
    parts = []

    # Engineer's query
    parts.append(f"## Engineer's Query\n{state.get('query', 'N/A')}")

    # Retrieved incidents
    incidents = state.get("retrieved_incidents", [])
    if incidents:
        parts.append("## Retrieved Historical Incidents")
        for inc in incidents[:5]:  # cap at 5 to control context length
            meta = inc.get("metadata", {})
            parts.append(
                f"**{inc['doc_id']}** [{meta.get('severity','?').upper()}] "
                f"{meta.get('region','?')} | {meta.get('equipment_type','?')} | "
                f"{meta.get('outage_event','?')}\n{inc['document'][:400]}"
            )

    # Stability telemetry summary
    label = state.get("stability_label")
    if label:
        parts.append(
            f"## Real-Time Stability Summary\n"
            f"Label: {label} | Probability unstable: {state.get('stability_prob', 'N/A'):.3f} | "
            f"Stab score: {state.get('stab_score', 'N/A'):.4f} | "
            f"Health score: {state.get('health_score', 'N/A')}/100 | "
            f"Grid frequency status: {state.get('grid_freq_status', 'N/A')}"
        )
        anomalies = state.get("anomaly_flags", [])
        if anomalies:
            parts.append(f"Anomalies detected: {len(anomalies)} events. "
                         f"Most critical: {anomalies[0].get('transformer_status','?')} "
                         f"at {anomalies[0].get('timestamp','?')}")

    # SHAP top 5 features
    shap = state.get("shap_top5", [])
    if shap:
        shap_text = " | ".join(
            f"{s['feature']}={s['value']:.3f} (SHAP {s['shap_value']:+.3f})"
            for s in shap
        )
        parts.append(f"## Top SHAP Features (Instability Drivers)\n{shap_text}")

    # Smart meter context (if available)
    sm_summary = state.get("smart_meter_summary")
    if sm_summary:
        parts.append(f"## Smart Meter Analysis\n{sm_summary}")

    return "\n\n".join(parts)


def analyse_failure(state: AgentState) -> AgentState:
    """
    LangGraph node: Failure Analysis Agent.

    Reads:  retrieved_incidents, stability_label, shap_top5, anomaly_flags,
            smart_meter_summary, query
    Writes: state.root_cause
    """
    request_id = state.get("request_id", "N/A")
    log_pipeline_event(request_id, "Failure Analysis Agent", "start", {})
    logger.info(f"[{request_id}] Failure Analysis Agent starting.")

    context = _build_context(state)
    llm     = get_model_router().get_llm("complex", temperature=0.1)

    try:
        messages = [
            SystemMessage(content=FAILURE_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response  = llm.invoke(messages)
        raw       = response.content.strip()
        root_cause: Dict[str, Any] = json.loads(raw)

        logger.info(
            f"[{request_id}] Root cause: {root_cause.get('probable_cause', 'N/A')[:80]} "
            f"(confidence={root_cause.get('confidence', 0):.2f})"
        )
        log_pipeline_event(
            request_id, "Failure Analysis Agent", "complete",
            {
                "probable_cause":  root_cause.get("probable_cause", "")[:100],
                "confidence":      root_cause.get("confidence", 0),
                "related_ids":     root_cause.get("related_incident_ids", []),
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Failure analysis LLM error: {e}")
        root_cause = {
            "probable_cause":       "Analysis failed — LLM error.",
            "evidence":             str(e),
            "confidence":           0.0,
            "related_incident_ids": [],
        }
        log_pipeline_event(request_id, "Failure Analysis Agent", "error", {"error": str(e)})

    return {**state, "root_cause": root_cause}
