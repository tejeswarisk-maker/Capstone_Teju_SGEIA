"""
retrieval_agent.py — Grid Retrieval Agent node.

Performs hybrid search (BM25 + ChromaDB semantic) over the incidents corpus,
fuses results via RRF, then reranks with a Cross-Encoder for final top-K.

Pattern: Day 4 Hybrid RAG + BM25 + RRF + Reranking notebooks.
"""

from typing import List

from src.agents.state import AgentState, RetrievedIncident
from src.indexing.bm25_index import get_bm25_index, rrf_fuse
from src.indexing.chroma_store import get_chroma_store
from src.config import settings
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)


def _build_chroma_filter(metadata_filters: dict) -> dict | None:
    """
    Convert flat metadata_filters dict to ChromaDB where-filter format.
    Supports single or multi-field AND filters.
    """
    if not metadata_filters:
        return None
    if len(metadata_filters) == 1:
        k, v = next(iter(metadata_filters.items()))
        return {k: v}
    # Multiple filters → wrap in $and
    return {"$and": [{k: v} for k, v in metadata_filters.items()]}


def _rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]:
    """
    Cross-Encoder reranking of hybrid-fused candidates.
    Uses ms-marco-MiniLM-L-6-v2 locally (no API cost).
    Falls back to returning candidates as-is on import error.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["document"]) for c in candidates]
        scores = model.predict(pairs)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Cross-Encoder reranking failed: {e} — returning RRF order.")
        for i, c in enumerate(candidates[:top_k]):
            c["rerank_score"] = None
        return candidates[:top_k]


def retrieve_incidents(state: AgentState) -> AgentState:
    """
    LangGraph node: Grid Retrieval Agent.

    Reads:  state.query, state.metadata_filters
    Writes: state.retrieved_incidents, state.retrieval_method, state.retrieval_count
    """
    request_id       = state.get("request_id", "N/A")
    query            = state.get("query", "")
    metadata_filters = state.get("metadata_filters", {})

    log_pipeline_event(request_id, "Grid Retrieval Agent", "start",
                       {"query_preview": query[:80], "filters": metadata_filters})
    logger.info(f"[{request_id}] Starting hybrid retrieval for: '{query[:60]}...'")

    chroma_filter = _build_chroma_filter(metadata_filters)
    candidate_k   = settings.retrieval_candidate_k
    final_k       = settings.retrieval_final_k

    # ── 1. BM25 keyword search ────────────────────────────────────────────────
    bm25_idx = get_bm25_index()
    bm25_results = bm25_idx.search(query, k=candidate_k)
    logger.debug(f"[{request_id}] BM25 returned {len(bm25_results)} candidates.")

    # ── 2. ChromaDB semantic search ───────────────────────────────────────────
    store = get_chroma_store()
    chroma_results = store.search_incidents(
        query=query,
        n_results=candidate_k,
        filters=chroma_filter,
    )
    logger.debug(f"[{request_id}] ChromaDB returned {len(chroma_results)} candidates.")

    # ── 3. RRF fusion ─────────────────────────────────────────────────────────
    fused = rrf_fuse(bm25_results, chroma_results, final_k=candidate_k)
    logger.info(f"[{request_id}] RRF fusion: {len(fused)} candidates after merge.")

    # ── 4. Cross-Encoder reranking ────────────────────────────────────────────
    reranked = _rerank(query, fused, top_k=final_k)
    logger.info(f"[{request_id}] Reranked to top-{len(reranked)} incidents.")

    # ── 5. Build RetrievedIncident list ──────────────────────────────────────
    retrieved: List[RetrievedIncident] = [
        RetrievedIncident(
            doc_id=c.get("doc_id", c.get("id", f"doc_{i}")),
            document=c["document"],
            metadata=c.get("metadata", {}),
            rrf_score=c.get("rrf_score", 0.0),
            rerank_score=c.get("rerank_score"),
        )
        for i, c in enumerate(reranked)
    ]

    log_pipeline_event(
        request_id, "Grid Retrieval Agent", "complete",
        {
            "bm25_hits":    len(bm25_results),
            "chroma_hits":  len(chroma_results),
            "fused_hits":   len(fused),
            "final_hits":   len(retrieved),
            "top_doc":      retrieved[0]["doc_id"] if retrieved else None,
        },
    )

    return {
        **state,
        "retrieved_incidents": retrieved,
        "retrieval_method":    "hybrid_bm25_chroma_rrf_crossencoder",
        "retrieval_count":     len(retrieved),
    }
