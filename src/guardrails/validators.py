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
    # Core grid terms
    "grid", "power", "energy", "voltage", "current", "transformer", "substation",
    "outage", "stability", "frequency", "meter", "smart meter", "demand", "load",
    "watt", "kwh", "ampere", "circuit", "fault", "trip", "overload", "distribution",
    "transmission", "feeder", "relay", "scada", "renewable", "solar", "wind",
    "zone", "region", "incident", "anomaly", "deviation", "excursion", "failure",
    "restoration", "equipment", "infrastructure", "forecast", "consumption", "peak",
    # Widget / dashboard context terms
    "health", "score", "critical", "high", "medium", "low", "severity", "risk",
    "mitigation", "recommendation", "analysis", "trend", "monitor", "detect",
    "explain", "summarise", "summarize", "compare", "show", "why", "what", "how",
    "rate", "count", "status", "feed", "agent", "bank", "cluster", "phase",
    "cascade", "instability", "dip", "spike", "drop", "imbalance", "safe", "limit",
    "INC", "AN-", "zone_a", "zone_b", "zone_c", "zone_d", "sm-c", "t-22",
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


# ── Harmful / violent content patterns ────────────────────────────────────────
HARMFUL_PATTERNS = [
    # Violence / threats
    r'\b(kill|murder|stab|shoot|bomb|attack|harm|hurt|threaten|assassin|terror)\b',
    # Self-harm
    r'\b(suicide|self.?harm|cut myself|end my life)\b',
    # Weapons
    r'\b(knife|gun|weapon|explosive|grenade|poison)\b.*\b(person|people|human|someone|him|her)\b',
    # Illegal
    r'\b(hack|ddos|malware|ransomware|phish|steal data|illegal)\b',
    # Sexual / inappropriate
    r'\b(sex|porn|naked|nude|rape|molest)\b',
]


def _check_harmful(query: str) -> Optional[str]:
    """
    Return rejection reason if query contains harmful/violent/illegal content.
    """
    import re as _re
    lower = query.lower()
    for pattern in HARMFUL_PATTERNS:
        if _re.search(pattern, lower):
            return (
                "⚠️ This query contains content that cannot be processed by SGEIA. "
                "SGEIA is a Smart Grid Energy Intelligence Assistant and only handles "
                "questions about power grid operations, stability, incidents, and energy systems."
            )
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
    Four-layer validation pipeline:
      1. Format check     — length, encoding
      2. Harmful content  — violence, threats, illegal, self-harm
      3. Domain relevance — must relate to grid / energy / datasets
      4. PII masking      — Presidio + regex fallback
    """
    # ── Layer 1: Format ───────────────────────────────────────────────────────
    format_error = _check_format(raw_query)
    if format_error:
        logger.warning(f"Query rejected (format): {format_error}")
        return ValidationResult(is_valid=False, rejection_reason=format_error)

    # ── Layer 2: Harmful content (checked FIRST — highest priority) ───────────
    harmful_reason = _check_harmful(raw_query)
    if harmful_reason:
        logger.warning(f"Query rejected (harmful): '{raw_query[:60]}'")
        return ValidationResult(is_valid=False, rejection_reason=harmful_reason)

    # ── Layer 3: Domain relevance ─────────────────────────────────────────────
    if not _check_domain_relevance(raw_query):
        reason = (
            "🚫 This query is outside SGEIA's scope.\n\n"
            "SGEIA only handles questions about:\n"
            "• Grid stability and health scores\n"
            "• Incidents, outages, and zone analysis\n"
            "• Smart meter consumption and anomalies\n"
            "• Voltage, frequency, transformer health\n"
            "• DS1 / DS2 dataset queries\n\n"
            "Please rephrase your question in the context of smart grid operations."
        )
        logger.warning(f"Query rejected (out-of-domain): '{raw_query[:60]}'")
        return ValidationResult(is_valid=False, rejection_reason=reason)

    # ── Layer 4: PII masking ──────────────────────────────────────────────────
    sanitised, pii_detected, pii_entities = _mask_pii_presidio(raw_query)

    # Additional regex-based masking (IDs, account numbers, coordinates)
    import re as _re
    sanitised = _re.sub(r'\b\d{10,}\b', '[ID_MASKED]', sanitised)          # long numeric IDs
    sanitised = _re.sub(r'\b[A-Z]{2,}\d{4,}\b', '[REF_MASKED]', sanitised) # ref codes like INC0042
    sanitised = _re.sub(r'\b\d{1,3}\.\d+[NS],?\s*\d{1,3}\.\d+[EW]\b',
                        '[GPS_MASKED]', sanitised)                           # GPS coordinates

    if pii_detected:
        logger.info(f"PII masked: {pii_entities}")

    return ValidationResult(
        is_valid=True,
        sanitised_query=sanitised.strip(),
        pii_detected=pii_detected,
        pii_entities=pii_entities,
    )
