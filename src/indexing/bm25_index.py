"""
bm25_index.py — BM25 keyword index for hybrid search over the incidents corpus.

Built once at startup from grid_incidents_synthetic.csv and kept in memory.
Combined with ChromaDB semantic search via Reciprocal Rank Fusion (RRF).

Pattern: Day 4 Hybrid RAG + BM25 + RRF notebook.

Usage:
    from src.indexing.bm25_index import BM25Index
    idx = BM25Index()
    idx.build()
    results = idx.search("transformer overload Zone_B", k=20)
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import INCIDENTS_CSV
from src.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokeniser used for both indexing and querying."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class BM25Index:
    """
    BM25 index over the incidents corpus.

    The index is built on the full text representation of each incident:
    description + key metadata fields concatenated to improve keyword recall
    on operational terms like zone names, equipment types, and event labels.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._docs: List[Dict[str, Any]] = []  # original incident rows
        self._corpus_tokens: List[List[str]] = []

    def build(self, force: bool = False) -> int:
        """
        Build the BM25 index from the incidents CSV.

        Args:
            force: Rebuild even if already built (useful after re-indexing).

        Returns:
            Number of documents indexed.
        """
        if self._bm25 is not None and not force:
            logger.debug("BM25 index already built — skipping.")
            return len(self._docs)

        if not INCIDENTS_CSV.exists():
            logger.error(f"Incidents CSV not found at {INCIDENTS_CSV}. Run generate_incidents first.")
            return 0

        df = pd.read_csv(INCIDENTS_CSV)
        logger.info(f"Building BM25 index over {len(df)} incidents...")

        self._docs = []
        self._corpus_tokens = []

        for _, row in df.iterrows():
            # Combine description + metadata text for richer keyword matching
            full_text = (
                f"{row['description']} "
                f"{row['region']} {row['equipment_type']} "
                f"{row['outage_event']} {row['severity']} "
                f"{row['transformer_status']}"
            )
            tokens = _tokenize(full_text)
            self._corpus_tokens.append(tokens)
            self._docs.append(row.to_dict())

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(
            f"BM25 index built: {len(self._docs)} docs, "
            f"vocab size: {len(self._bm25.idf)} terms."
        )
        return len(self._docs)

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Keyword search using BM25.

        Args:
            query: Natural-language or keyword query string.
            k:     Number of top results to return.

        Returns:
            List of dicts with keys: rank, doc_id, bm25_score, document, metadata.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built — call build() first.")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for rank, (idx, score) in enumerate(ranked):
            doc = self._docs[idx]
            results.append({
                "rank":      rank + 1,
                "doc_id":    doc.get("incident_id", f"doc_{idx}"),
                "bm25_score": round(float(score), 4),
                "document":  doc.get("description", ""),
                "metadata": {
                    "region":            doc.get("region"),
                    "equipment_type":    doc.get("equipment_type"),
                    "outage_event":      doc.get("outage_event"),
                    "severity":          doc.get("severity"),
                    "transformer_status": doc.get("transformer_status"),
                    "timestamp":         doc.get("timestamp"),
                },
            })
        return results


def rrf_fuse(
    bm25_results: List[Dict],
    chroma_results: List[Dict],
    k_rrf: int = 60,
    final_k: int = 20,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion — combines BM25 and ChromaDB ranked lists.

    Formula: RRF(d) = Σ 1/(k_rrf + rank_i(d))

    Args:
        bm25_results:   Results from BM25Index.search()
        chroma_results: Results from ChromaStore.search_incidents()
        k_rrf:          RRF smoothing constant (default 60 per original paper)
        final_k:        Number of fused results to return

    Returns:
        Fused and re-ranked list of dicts, each with an 'rrf_score' key.
    """
    scores: Dict[str, float] = {}
    docs:   Dict[str, Dict]  = {}

    # Score BM25 results
    for item in bm25_results:
        doc_id = item["doc_id"]
        rank   = item["rank"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        docs[doc_id]   = item

    # Score ChromaDB results (use position in list as rank)
    for rank, item in enumerate(chroma_results, start=1):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id":   doc_id,
                "document": item["document"],
                "metadata": item["metadata"],
                "bm25_score": 0.0,
            }

    # Sort by fused RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    fused = []
    for new_rank, (doc_id, rrf_score) in enumerate(ranked, start=1):
        entry = dict(docs[doc_id])
        entry["rank"]      = new_rank
        entry["rrf_score"] = round(rrf_score, 6)
        fused.append(entry)

    return fused


# ── Module-level singleton ─────────────────────────────────────────────────────
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return the shared BM25Index singleton (builds on first access)."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        _bm25_index.build()
    return _bm25_index
