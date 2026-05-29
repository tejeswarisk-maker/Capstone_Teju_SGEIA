"""
deepeval_suite.py — SGEIA DeepEval Evaluation Suite.

Runs a set of LLM evaluation metrics against the full agent pipeline output to
measure response quality, faithfulness, hallucination, and answer relevance.

Metrics used:
  - FaithfulnessMetric    : Are mitigation steps grounded in retrieved context?
  - HallucinationMetric   : Does the response introduce unsupported facts?
  - AnswerRelevancyMetric : Is the final response relevant to the input query?
  - GEval (custom)        : Custom criteria for grid-domain technical quality

Usage:
    # Run full suite against live API
    python -m src.evaluation.deepeval_suite

    # Run in dry-run mode (no API calls — uses cached fixtures)
    python -m src.evaluation.deepeval_suite --dry-run

    # Run specific metric only
    python -m src.evaluation.deepeval_suite --metric faithfulness

Outputs:
  - Console table with pass/fail per test case
  - logs/eval_report_{timestamp}.json — JSON export of all results
"""

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import get_logger, log_pipeline_event
from src.config import settings

logger = get_logger(__name__)

# ── Output directory ───────────────────────────────────────────────────────────
EVAL_LOG_DIR = Path("logs")
EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Evaluation thresholds ─────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD   = settings.faithfulness_threshold   # 0.7
HALLUCINATION_THRESHOLD  = 0.3   # below this = acceptable hallucination risk
RELEVANCY_THRESHOLD      = 0.7
GEVAL_THRESHOLD          = 0.6


# ── Test fixtures — representative grid queries with expected behaviour ────────

TEST_CASES = [
    {
        "id":             "TC-EVAL-001",
        "query":          "Analyse voltage instability in Zone_B transformers",
        "expected_route": "incident",
        "description":    "Incident retrieval + root cause for zone-specific voltage fault",
    },
    {
        "id":             "TC-EVAL-002",
        "query":          "What is the current grid stability and health score?",
        "expected_route": "stability",
        "description":    "Stability assessment with health score and frequency status",
    },
    {
        "id":             "TC-EVAL-003",
        "query":          "Detect smart meter consumption anomalies",
        "expected_route": "smart_meter",
        "description":    "Smart meter anomaly detection from DS2",
    },
    {
        "id":             "TC-EVAL-004",
        "query":          "Find similar incidents to Zone_C partial outage with critical transformer status",
        "expected_route": "incident",
        "description":    "Semantic incident search with metadata filter",
    },
    {
        "id":             "TC-EVAL-005",
        "query":          "Why did the grid frequency deviate last week and what should we do?",
        "expected_route": "cross_domain",
        "description":    "Cross-domain: stability + incidents + mitigation",
    },
    {
        "id":             "TC-EVAL-006",
        "query":          "Explain transformer overload causes and provide mitigation in Zone_A",
        "expected_route": "incident",
        "description":    "Root cause + mitigation for transformer overload",
    },
    {
        "id":             "TC-EVAL-007",
        "query":          "What is the demand forecast and anomaly count for smart meters?",
        "expected_route": "smart_meter",
        "description":    "Smart meter demand forecast evaluation",
    },
    {
        "id":             "TC-EVAL-008",
        "query":          "Full grid assessment: stability, incidents, and smart meter status",
        "expected_route": "cross_domain",
        "description":    "Comprehensive cross-domain assessment",
    },
]

# ── GEval criteria for grid-domain technical quality ─────────────────────────
GEVAL_CRITERIA = """
Evaluate the response on the following criteria for a power grid intelligence system:

1. Technical accuracy: Does the response use correct power grid terminology?
2. Actionability: Are the recommendations specific and implementable by a grid operator?
3. Evidence grounding: Does the analysis cite or reference specific incident data?
4. Completeness: Does the response address the original question fully?
5. Safety awareness: Does the response prioritise safety-critical actions first?

Score 0-1 where 1 = excellent on all criteria.
"""


# ── Core evaluation runner ────────────────────────────────────────────────────

def _call_pipeline(query: str) -> Dict[str, Any]:
    """
    Call the agent pipeline directly (bypasses HTTP — imports run_query).
    Returns the full AgentState result dict.
    """
    from src.agents.graph import run_query
    session_id = str(uuid.uuid4())
    return run_query(query=query, session_id=session_id)


def _build_deepeval_test_case(
    query: str,
    result: Dict[str, Any],
):
    """
    Construct a DeepEval LLMTestCase from pipeline result.

    Args:
        query:  Original user query string.
        result: AgentState dict returned by run_query.

    Returns:
        deepeval.test_case.LLMTestCase instance.
    """
    from deepeval.test_case import LLMTestCase

    actual_output = result.get("final_response", "")
    retrieval_context = [
        inc["document"]
        for inc in (result.get("retrieved_incidents") or [])
    ]

    # Build expected output hint from mitigation steps (for relevancy check)
    steps = result.get("mitigation_steps") or []
    expected_output = "\n".join(steps) if steps else actual_output[:200]

    return LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )


def _run_faithfulness(test_case, threshold: float) -> Dict[str, Any]:
    """Run DeepEval FaithfulnessMetric."""
    try:
        from deepeval.metrics import FaithfulnessMetric
        metric = FaithfulnessMetric(threshold=threshold, verbose_mode=False)
        metric.measure(test_case)
        return {
            "metric":    "faithfulness",
            "score":     metric.score,
            "passed":    metric.is_successful(),
            "reason":    metric.reason,
        }
    except Exception as e:
        logger.warning(f"FaithfulnessMetric failed: {e}")
        return {"metric": "faithfulness", "score": None, "passed": None, "reason": str(e)}


def _run_hallucination(test_case, threshold: float) -> Dict[str, Any]:
    """Run DeepEval HallucinationMetric."""
    try:
        from deepeval.metrics import HallucinationMetric
        metric = HallucinationMetric(threshold=threshold, verbose_mode=False)
        metric.measure(test_case)
        return {
            "metric":    "hallucination",
            "score":     metric.score,
            "passed":    metric.is_successful(),
            "reason":    metric.reason,
        }
    except Exception as e:
        logger.warning(f"HallucinationMetric failed: {e}")
        return {"metric": "hallucination", "score": None, "passed": None, "reason": str(e)}


def _run_answer_relevancy(test_case, threshold: float) -> Dict[str, Any]:
    """Run DeepEval AnswerRelevancyMetric."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        metric = AnswerRelevancyMetric(threshold=threshold, verbose_mode=False)
        metric.measure(test_case)
        return {
            "metric":    "answer_relevancy",
            "score":     metric.score,
            "passed":    metric.is_successful(),
            "reason":    metric.reason,
        }
    except Exception as e:
        logger.warning(f"AnswerRelevancyMetric failed: {e}")
        return {"metric": "answer_relevancy", "score": None, "passed": None, "reason": str(e)}


def _run_geval(test_case, threshold: float) -> Dict[str, Any]:
    """Run DeepEval GEval with custom grid-domain criteria."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams
        metric = GEval(
            name="GridDomainQuality",
            criteria=GEVAL_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=threshold,
            verbose_mode=False,
        )
        metric.measure(test_case)
        return {
            "metric":    "geval_grid_quality",
            "score":     metric.score,
            "passed":    metric.is_successful(),
            "reason":    metric.reason,
        }
    except Exception as e:
        logger.warning(f"GEval failed: {e}")
        return {"metric": "geval_grid_quality", "score": None, "passed": None, "reason": str(e)}


def _check_routing(result: Dict, expected_route: str) -> Dict[str, Any]:
    """
    Verify the routing decision matches expected routing.
    This is a deterministic check, not an LLM metric.
    """
    actual_route = result.get("routing_decision", "")
    passed = actual_route == expected_route
    return {
        "metric":    "routing_accuracy",
        "score":     1.0 if passed else 0.0,
        "passed":    passed,
        "reason":    f"Expected '{expected_route}', got '{actual_route}'",
    }


# ── Single test case evaluator ────────────────────────────────────────────────

def evaluate_test_case(
    tc: Dict[str, Any],
    metrics: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run all requested metrics against a single test case.

    Args:
        tc:      Test case dict with id, query, expected_route, description.
        metrics: List of metric names to run ('faithfulness', 'hallucination',
                 'relevancy', 'geval', 'routing').
        dry_run: If True, skip pipeline call and use stub results.

    Returns:
        Dict with test case metadata + list of metric results.
    """
    tc_id   = tc["id"]
    query   = tc["query"]
    logger.info(f"Evaluating {tc_id}: '{query[:60]}'")

    eval_record: Dict[str, Any] = {
        "test_case_id":   tc_id,
        "query":          query,
        "description":    tc["description"],
        "expected_route": tc["expected_route"],
        "timestamp":      datetime.utcnow().isoformat(),
        "metrics":        [],
        "pipeline_error": None,
    }

    # ── Pipeline call ──────────────────────────────────────────────────────────
    if dry_run:
        logger.info(f"[{tc_id}] Dry-run mode — skipping pipeline call.")
        result = {
            "final_response":      f"[DRY RUN] Stub response for: {query}",
            "routing_decision":    tc["expected_route"],
            "retrieved_incidents": [],
            "mitigation_steps":    [],
        }
    else:
        try:
            t0 = time.time()
            result = _call_pipeline(query)
            elapsed = round(time.time() - t0, 2)
            eval_record["pipeline_elapsed_s"] = elapsed
            logger.info(f"[{tc_id}] Pipeline completed in {elapsed}s")
        except Exception as e:
            logger.error(f"[{tc_id}] Pipeline call failed: {e}")
            eval_record["pipeline_error"] = str(e)
            return eval_record

    eval_record["routing_decision"] = result.get("routing_decision")
    eval_record["health_score"]     = result.get("health_score")

    # ── Build DeepEval test case ───────────────────────────────────────────────
    try:
        deepeval_tc = _build_deepeval_test_case(query, result)
    except Exception as e:
        logger.error(f"[{tc_id}] Failed to build DeepEval test case: {e}")
        eval_record["pipeline_error"] = f"test_case_build: {e}"
        return eval_record

    # ── Run requested metrics ──────────────────────────────────────────────────
    metric_results = []

    if "routing" in metrics:
        metric_results.append(_check_routing(result, tc["expected_route"]))

    if "faithfulness" in metrics and result.get("retrieved_incidents"):
        metric_results.append(_run_faithfulness(deepeval_tc, FAITHFULNESS_THRESHOLD))

    if "hallucination" in metrics and result.get("retrieved_incidents"):
        metric_results.append(_run_hallucination(deepeval_tc, HALLUCINATION_THRESHOLD))

    if "relevancy" in metrics:
        metric_results.append(_run_answer_relevancy(deepeval_tc, RELEVANCY_THRESHOLD))

    if "geval" in metrics:
        metric_results.append(_run_geval(deepeval_tc, GEVAL_THRESHOLD))

    eval_record["metrics"] = metric_results

    # Overall pass/fail: all non-None metrics must pass
    passed_metrics = [m for m in metric_results if m["passed"] is not None]
    eval_record["overall_pass"] = all(m["passed"] for m in passed_metrics) if passed_metrics else None

    return eval_record


# ── Full suite runner ─────────────────────────────────────────────────────────

def run_full_suite(
    metrics: Optional[List[str]] = None,
    dry_run: bool = False,
    test_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite across all test cases.

    Args:
        metrics:  Metrics to evaluate. Defaults to all metrics.
        dry_run:  Skip pipeline calls (structural test only).
        test_ids: Subset of test case IDs to run. If None, runs all.

    Returns:
        Summary dict with per-case results and aggregate stats.
    """
    if metrics is None:
        metrics = ["routing", "faithfulness", "hallucination", "relevancy", "geval"]

    run_id    = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()

    logger.info(f"[{run_id}] Starting SGEIA DeepEval suite | metrics={metrics} | dry_run={dry_run}")
    log_pipeline_event(run_id, "EvalSuite", "start",
                       {"metrics": metrics, "dry_run": dry_run, "n_cases": len(TEST_CASES)})

    cases_to_run = TEST_CASES
    if test_ids:
        cases_to_run = [tc for tc in TEST_CASES if tc["id"] in test_ids]

    all_results = []
    for tc in cases_to_run:
        record = evaluate_test_case(tc, metrics, dry_run=dry_run)
        all_results.append(record)
        _print_case_summary(record)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    total   = len(all_results)
    passed  = sum(1 for r in all_results if r.get("overall_pass") is True)
    failed  = sum(1 for r in all_results if r.get("overall_pass") is False)
    errored = sum(1 for r in all_results if r.get("pipeline_error"))

    # Per-metric averages
    metric_scores: Dict[str, List[float]] = {}
    for record in all_results:
        for m in record.get("metrics", []):
            name  = m["metric"]
            score = m.get("score")
            if score is not None:
                metric_scores.setdefault(name, []).append(score)

    metric_averages = {
        name: round(sum(scores) / len(scores), 4)
        for name, scores in metric_scores.items()
    }

    summary = {
        "run_id":          run_id,
        "timestamp":       timestamp,
        "dry_run":         dry_run,
        "metrics_run":     metrics,
        "total_cases":     total,
        "passed":          passed,
        "failed":          failed,
        "errored":         errored,
        "pass_rate":       round(passed / total, 4) if total else 0,
        "metric_averages": metric_averages,
        "results":         all_results,
    }

    # ── Write report ───────────────────────────────────────────────────────────
    report_path = EVAL_LOG_DIR / f"eval_report_{run_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"[{run_id}] Evaluation report saved → {report_path}")

    _print_suite_summary(summary)
    log_pipeline_event(run_id, "EvalSuite", "complete",
                       {"passed": passed, "failed": failed, "pass_rate": summary["pass_rate"]})

    return summary


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_case_summary(record: Dict[str, Any]) -> None:
    """Print a single test case result line."""
    tc_id   = record["test_case_id"]
    overall = record.get("overall_pass")
    icon    = "✅" if overall else ("❌" if overall is False else "⚠️")
    error   = f" [ERROR: {record['pipeline_error'][:60]}]" if record.get("pipeline_error") else ""
    scores  = " | ".join(
        f"{m['metric']}={m['score']:.2f}" if m.get('score') is not None else f"{m['metric']}=N/A"
        for m in record.get("metrics", [])
    )
    print(f"  {icon} {tc_id}: {record['description'][:50]:<50}  [{scores}]{error}")


def _print_suite_summary(summary: Dict[str, Any]) -> None:
    """Print overall suite summary."""
    print("\n" + "=" * 70)
    print(f"SGEIA DeepEval Suite — Run {summary['run_id']}")
    print(f"{'=' * 70}")
    print(f"  Total cases : {summary['total_cases']}")
    print(f"  Passed      : {summary['passed']}")
    print(f"  Failed      : {summary['failed']}")
    print(f"  Errored     : {summary['errored']}")
    print(f"  Pass rate   : {summary['pass_rate']:.1%}")
    if summary["metric_averages"]:
        print("\n  Metric Averages:")
        for name, avg in summary["metric_averages"].items():
            bar = "█" * int(avg * 20)
            print(f"    {name:<25} {avg:.3f}  {bar}")
    print("=" * 70)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SGEIA DeepEval Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip pipeline calls; test evaluation harness structure only",
    )
    parser.add_argument(
        "--metric",
        choices=["faithfulness", "hallucination", "relevancy", "geval", "routing", "all"],
        default="all",
        help="Metric(s) to run (use 'all' for every metric)",
    )
    parser.add_argument(
        "--test-ids", nargs="+", default=None,
        help="Specific test case IDs to run (e.g. TC-EVAL-001 TC-EVAL-003)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metrics = None if args.metric == "all" else [args.metric]
    run_full_suite(metrics=metrics, dry_run=args.dry_run, test_ids=args.test_ids)
