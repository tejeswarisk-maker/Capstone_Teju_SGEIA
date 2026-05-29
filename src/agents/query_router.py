"""
query_router.py — LLM Query Router node.

Classifies the user's query intent and extracts metadata filters.
Uses gpt-4o-mini (primary) with Gemini Flash fallback.
Outputs a routing_decision that determines which agents are invoked next.

Routing decisions:
  "incident"     → Grid Retrieval Agent only
  "stability"    → Grid Stability Agent + Failure Analysis Agent
  "smart_meter"  → Smart Meter Agent
  "cross_domain" → All agents (fan-out)
  "out_of_scope" → Guardrail rejection
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.models.model_router import get_model_router
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an AI routing assistant for a Smart Grid Energy Intelligence System.

Analyse the user query and return a JSON object with these exact keys:

{
  "routing_decision": "<one of: incident | stability | smart_meter | cross_domain | out_of_scope>",
  "extracted_intent": "<one-sentence description of what the user wants>",
  "metadata_filters": {
    "region": "<Zone_A|Zone_B|Zone_C|Zone_D or null if not mentioned>",
    "severity": "<critical|high|medium|low or null>",
    "equipment_type": "<transformer|distribution_unit|substation|smart_meter_bank|transmission_line|renewable_inverter or null>",
    "outage_event": "<full_outage|partial_outage|voltage_deviation|frequency_excursion|high_demand_event|meter_dropout|no_event|renewable_variability or null>"
  }
}

Routing rules:
- "incident"     → user asks about specific past events, incidents, or similar outages
- "stability"    → user asks about current stability, instability risk, grid health scores, anomalies
- "smart_meter"  → user asks about consumption patterns, meter readings, household demand
- "cross_domain" → query spans multiple domains or asks for overall grid picture
- "out_of_scope" → query is not about power grids, energy systems, or related infrastructure

Return only the JSON object. No markdown, no explanation."""


def route_query(state: AgentState) -> AgentState:
    """
    LangGraph node: LLM Query Router.

    Reads state.query and state.widget_context.
    Writes state.routing_decision, state.metadata_filters, state.extracted_intent.
    """
    request_id = state.get("request_id", "N/A")
    query      = state.get("query", "")
    widget_ctx = state.get("widget_context")

    log_pipeline_event(request_id, "LLM Query Router", "start", {"query_preview": query[:100]})
    logger.info(f"[{request_id}] Routing query: '{query[:80]}...'")

    # Enrich query with widget context if available
    full_query = query
    if widget_ctx:
        full_query = (
            f"{query}\n\n[Dashboard context — {widget_ctx.get('panel', 'unknown panel')}: "
            f"{json.dumps(widget_ctx.get('data', {}))[:300]}]"
        )

    # Call LLM router
    llm = get_model_router().get_llm("simple", temperature=0.0)

    try:
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=full_query),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Parse JSON response
        parsed: Dict[str, Any] = json.loads(raw)

        routing_decision = parsed.get("routing_decision", "cross_domain")
        extracted_intent = parsed.get("extracted_intent", "")
        metadata_filters = {
            k: v
            for k, v in parsed.get("metadata_filters", {}).items()
            if v is not None
        }

        logger.info(
            f"[{request_id}] Routing decision: {routing_decision} | "
            f"Intent: {extracted_intent} | Filters: {metadata_filters}"
        )
        log_pipeline_event(
            request_id, "LLM Query Router", "complete",
            {
                "routing_decision": routing_decision,
                "extracted_intent": extracted_intent,
                "metadata_filters": metadata_filters,
            },
        )

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[{request_id}] Router LLM failed: {e} — defaulting to cross_domain.")
        routing_decision = "cross_domain"
        extracted_intent = query
        metadata_filters = {}
        log_pipeline_event(request_id, "LLM Query Router", "error", {"error": str(e)})

    return {
        **state,
        "routing_decision": routing_decision,
        "extracted_intent": extracted_intent,
        "metadata_filters": metadata_filters,
    }
