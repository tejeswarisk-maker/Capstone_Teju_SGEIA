"""
validators.py — Input validation guardrails for the SGEIA API gateway.

Three-layer validation pipeline:
  1. Format check   — length, encoding, basic sanity
  2. Domain check   — query must relate to power grid / energy topics
  3. PII masking    — Presidio strips GPS coordinates, person names, IDs

Pattern: Day 10 GuardRAils_And_PII_Demo notebook.

Usage:
    from src.guardrails.validators import validate_and_sanitise
    result = validate_and_sanitise(user_input)
    if result.is_valid:
        proceed(result.sanitised_query)
    else:
        return result.rejection_reason
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)

# ── Domain keywords — at least one must appear in a valid grid query ──────────
DOMAIN_KEYWORDS = [
    "grid", "power", "energy", "voltage", "current", "transformer", "substation",
    "outage", "stability", "frequency", "meter", "smart meter", "demand", "load",
    "watt", "kwh", "ampere", "circuit", "fault", "trip", "overload", "distribution",
    "transmission", "feeder", "relay", "scada", "renewable", "solar", "wind",
    "zone", "region", "incident", "anomaly", "deviation", "excursion", "failure",
    "restoration", "equipment", "infrastructure", "forecast", "consumption", "peak",
]

# Query length constraints
MIN_QUERY_LEN = 5
MAX_QUERY_LEN = 1000


@dataclass
class ValidationResult:
    """Result returned by the validation pipeline."""
    is_valid:         bool
    sanitised_query:  str = ""
    rejection_reason: Optional[str] = None
    pii_detected:     bool = False
    pii_entities:     list = field(default_factory=list)


def _check_format(query: str) -> Optional[str]:
    """
    Check basic format constraints.
    Returns an error string if invalid, None if OK.
    """
    if not query or not query.strip():
        return "Query cannot be empty."
    if len(query.strip()) < MIN_QUERY_LEN:
        return f"Query too short (minimum {MIN_QUERY_LEN} characters)."
    if len(query) > MAX_QUERY_LEN:
        return f"Query too long (maximum {MAX_QUERY_LEN} characters)."
    return None


def _check_domain_relevance(query: str) -> bool:
    """
    Return True if the query contains at least one domain keyword.
    Case-insensitive.
    """
    lower = query.lower()
    return any(kw in lower for kw in DOMAIN_KEYWORDS)


def _mask_pii_presidio(query: str) -> tuple[str, bool, list]:
    """
    Use Microsoft Presidio to detect and anonymise PII in the query.

    Returns:
        (masked_query, pii_detected, entity_list)

    Falls back to regex masking if Presidio is not available.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(text=query, language="en")
        if not results:
            return query, False, []

        anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
        entities   = [r.entity_type for r in results]
        logger.info(f"PII detected and masked: {entities}")
        return anonymized.text, True, entities

    except ImportError:
        # Fallback: simple regex masking for GPS coordinates and emails
        masked = re.sub(
            r"\b\d{1,3}\.\d+[NS]?\s*,?\s*\d{1,3}\.\d+[EW]?\b",
            "<GPS_MASKED>", query
        )
        masked = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "<EMAIL_MASKED>", masked
        )
        pii_detected = masked != query
        return masked, pii_detected, ["regex_fallback"] if pii_detected else []

    except Exception as e:
        logger.warning(f"Presidio PII masking failed: {e} — returning original query.")
        return query, False, []


def validate_and_sanitise(raw_query: str) -> ValidationResult:
    """
    Run the full three-layer validation pipeline on a raw user query.

    Args:
        raw_query: The raw string submitted by the user.

    Returns:
        ValidationResult with sanitised query or rejection details.
    """
    # ── Layer 1: Format check ──────────────────────────────────────────────────
    format_error = _check_format(raw_query)
    if format_error:
        logger.warning(f"Query rejected (format): {format_error}")
        return ValidationResult(is_valid=False, rejection_reason=format_error)

    # ── Layer 2: Domain relevance ─────────────────────────────────────────────
    if not _check_domain_relevance(raw_query):
        reason = (
            "Your query doesn't appear to be related to power grid or energy systems. "
            "Please ask about grid stability, outages, transformer health, smart meter data, "
            "or similar operational topics."
        )
        logger.warning(f"Query rejected (out-of-domain): '{raw_query[:60]}'")
        return ValidationResult(is_valid=False, rejection_reason=reason)

    # ── Layer 3: PII masking ──────────────────────────────────────────────────
    sanitised, pii_detected, pii_entities = _mask_pii_presidio(raw_query)

    if pii_detected:
        logger.info(f"PII masked in query. Entities: {pii_entities}")

    return ValidationResult(
        is_valid=True,
        sanitised_query=sanitised.strip(),
        pii_detected=pii_detected,
        pii_entities=pii_entities,
    )
