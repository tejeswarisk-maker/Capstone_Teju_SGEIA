"""
recommendation_agent.py — Recommendation Agent node.

Generates explainable mitigation steps grounded in retrieved incidents.
Quality gated by:
  1. DeepEval faithfulness check (retrieval_context vs recommendation)
  2. LLM-as-judge scoring (technical soundness, grounding, actionability)

If either gate fails (score below threshold), mitigation is regenerated
with a refined prompt (max 2 retries, tracked in state.regeneration_count).
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.models.model_router import get_model_router
from src.config import settings
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)

MAX_REGENERATIONS = 2  # max retry loops before returning best attempt

MITIGATION_SYSTEM_PROMPT = """You are a senior utility grid engineer providing operational mitigation guidance.

You are given the root cause analysis of a grid failure and supporting evidence from historical incidents.

Generate a clear, actionable mitigation plan with numbered steps.
Each step must be:
  - Technically specific (mention equipment, parameters, thresholds)
  - Grounded in the provided incident evidence (cite incident IDs where relevant)
  - Ordered by urgency (immediate actions first, preventive last)

Respond in JSON:
{
  "mitigation_steps": [
    "1. <immediate action>",
    "2. <short-term action>",
    "3. <medium-term action>",
    "4. <preventive measure>"
  ],
  "grounding_citations": ["INC-XXXX", ...],
  "confidence_note": "<one sentence on confidence level>"
}"""

JUDGE_SYSTEM_PROMPT = """You are a grid operations quality assessor.
Evaluate the mitigation plan on three criteria. Score each 1–5 (5=best).
Return JSON only:
{
  "technical_soundness": <int 1-5>,
  "grounding_in_evidence": <int 1-5>,
  "actionability": <int 1-5>,
  "reasoning": "<one sentence>"
}"""


def _build_mitigation_context(state: AgentState) -> str:
    """Assemble the context message for mitigation generation."""
    parts = []

    # Root cause
    rc = state.get("root_cause", {})
    parts.append(
        f"## Root Cause Analysis\n"
        f"Probable cause: {rc.get('probable_cause', 'Unknown')}\n"
        f"Evidence: {rc.get('evidence', 'N/A')}\n"
        f"Confidence: {rc.get('confidence', 0):.2f}"
    )

    # Supporting incidents
    incidents = state.get("retrieved_incidents", [])[:5]
    if incidents:
        parts.append("## Supporting Incident Records")
        for inc in incidents:
            meta = inc.get("metadata", {})
            parts.append(
                f"- **{inc['doc_id']}** ({meta.get('severity','?')}) "
                f"{meta.get('region','?')}/{meta.get('equipment_type','?')}: "
                f"{inc['document'][:300]}"
            )

    # Stability context
    if state.get("stability_label"):
        parts.append(
            f"## Current Grid State\n"
            f"Stability: {state.get('stability_label')} | "
            f"Health: {state.get('health_score', 'N/A')}/100 | "
            f"Frequency: {state.get('grid_freq_status', 'N/A')}"
        )

    # Smart meter
    if state.get("smart_meter_summary"):
        parts.append(f"## Smart Meter Context\n{state.get('smart_meter_summary')}")

    return "\n\n".join(parts)


def _generate_mitigation(state: AgentState, llm, attempt: int) -> Dict[str, Any]:
    """Call the LLM to generate mitigation steps."""
    context = _build_mitigation_context(state)
    if attempt > 0:
        context = (
            f"IMPORTANT: Previous mitigation attempt was rated below quality threshold. "
            f"Please provide more specific, evidence-grounded steps.\n\n{context}"
        )
    messages = [
        SystemMessage(content=MITIGATION_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    return json.loads(response.content.strip())


def _llm_judge(mitigation: Dict, root_cause: Dict, incidents: List, llm) -> float:
    """Run LLM-as-judge scoring. Returns average score (1–5)."""
    judge_context = (
        f"Root cause: {root_cause.get('probable_cause', 'N/A')}\n"
        f"Evidence: {root_cause.get('evidence', 'N/A')}\n"
        f"Mitigation steps:\n" + "\n".join(mitigation.get("mitigation_steps", []))
    )
    try:
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_context),
        ]
        resp = llm.invoke(messages)
        scores = json.loads(resp.content.strip())
        avg = (
            scores.get("technical_soundness", 3)
            + scores.get("grounding_in_evidence", 3)
            + scores.get("actionability", 3)
        ) / 3
        return round(avg, 2)
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 3.0  # neutral score on failure


def _deepeval_faithfulness(
    mitigation_steps: List[str],
    retrieval_context: List[str],
) -> float:
    """
    Run DeepEval faithfulness check.
    Returns score 0–1 (1 = fully grounded in retrieved context).
    Falls back to 1.0 if DeepEval is not available.
    """
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        actual_output = " ".join(mitigation_steps)
        test_case = LLMTestCase(
            input="Generate mitigation steps",
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        metric = FaithfulnessMetric(threshold=settings.faithfulness_threshold)
        metric.measure(test_case)
        return metric.score
    except Exception as e:
        logger.warning(f"DeepEval faithfulness check failed: {e} — skipping gate.")
        return 1.0  # pass through if unavailable


def generate_recommendations(state: AgentState) -> AgentState:
    """
    LangGraph node: Recommendation Agent.

    Reads:  root_cause, retrieved_incidents, stability state, smart_meter_summary
    Writes: mitigation_steps, judge_score, faithfulness_score, regeneration_count
    """
    request_id        = state.get("request_id", "N/A")
    regeneration_count = state.get("regeneration_count", 0)

    log_pipeline_event(request_id, "Recommendation Agent", "start",
                       {"attempt": regeneration_count + 1})
    logger.info(f"[{request_id}] Recommendation Agent (attempt {regeneration_count + 1}).")

    llm_complex = get_model_router().get_llm("complex", temperature=0.2)
    llm_simple  = get_model_router().get_llm("simple",  temperature=0.0)

    incidents = state.get("retrieved_incidents", [])
    retrieval_context = [inc["document"] for inc in incidents]

    best_mitigation: Optional[Dict] = None
    best_judge_score  = 0.0
    best_faith_score  = 0.0

    for attempt in range(MAX_REGENERATIONS + 1):
        try:
            # ── Generate mitigation ───────────────────────────────────────────
            mitigation = _generate_mitigation(state, llm_complex, attempt)
            steps = mitigation.get("mitigation_steps", [])

            if not steps:
                logger.warning(f"[{request_id}] Empty mitigation steps on attempt {attempt + 1}.")
                continue

            # ── Faithfulness gate ─────────────────────────────────────────────
            faith_score = _deepeval_faithfulness(steps, retrieval_context)
            logger.info(f"[{request_id}] Faithfulness score: {faith_score:.3f}")

            # ── LLM-as-judge gate ─────────────────────────────────────────────
            judge_score = _llm_judge(
                mitigation, state.get("root_cause", {}), incidents, llm_simple
            )
            logger.info(f"[{request_id}] Judge score: {judge_score:.2f}/5")

            if judge_score > best_judge_score:
                best_mitigation  = mitigation
                best_judge_score = judge_score
                best_faith_score = faith_score

            # Pass both gates → done
            if faith_score >= settings.faithfulness_threshold and judge_score >= settings.judge_score_threshold:
                logger.info(f"[{request_id}] Recommendation passed quality gates on attempt {attempt + 1}.")
                break
            else:
                logger.warning(
                    f"[{request_id}] Quality gate failed on attempt {attempt + 1} — "
                    f"faith={faith_score:.2f}, judge={judge_score:.2f}. Regenerating..."
                )

        except Exception as e:
            logger.error(f"[{request_id}] Recommendation generation error: {e}")
            break

    # Use best attempt even if gates not fully passed
    if best_mitigation is None:
        best_mitigation = {"mitigation_steps": ["Unable to generate recommendations — please retry."]}

    final_steps = best_mitigation.get("mitigation_steps", [])

    log_pipeline_event(
        request_id, "Recommendation Agent", "complete",
        {
            "steps_count":       len(final_steps),
            "judge_score":       best_judge_score,
            "faithfulness_score": best_faith_score,
            "total_attempts":    regeneration_count + 1,
        },
    )

    return {
        **state,
        "mitigation_steps":   final_steps,
        "judge_score":        best_judge_score,
        "faithfulness_score": best_faith_score,
        "regeneration_count": regeneration_count + len(final_steps),  # track total
    }
