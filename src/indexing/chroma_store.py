"""
chroma_store.py — ChromaDB vector store manager for all three pipelines.

Collections managed:
  'incidents'     — 200 synthetic incident narratives (primary RAG corpus)
  'grid_stability' — DS1 rows serialised as structured text
  'smart_meter'    — DS2 daily aggregates as text documents
  'query_cache'    — Cached query embeddings for deduplication

Each collection stores:
  - Embedded text (description / serialised row)
  - Metadata for filtering: region, equipment_type, outage_event, severity, etc.

Usage:
    from src.indexing.chroma_store import ChromaStore
    store = ChromaStore()
    store.index_incidents()           # run once
    results = store.search_incidents(query, filters={"region": "Zone_B"})
"""

import hashlib
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
from chromadb.config import Settings as ChromaSettings

from src.config import (
    CHROMA_DIR, INCIDENTS_CSV, DS1_AUGMENTED, DS2_AUGMENTED, settings
)
from src.models.model_router import get_model_router
from src.logger import get_logger, log_pipeline_event

logger = get_logger(__name__)


class ChromaStore:
    """
    Manages ChromaDB collections for all SGEIA pipelines.

    The client is persistent — data survives between runs.
    Embeddings are generated via the ModelRouter (OpenAI primary, local fallback).
    """

    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._router = get_model_router()
        self._embed_fn = None  # lazy init

    def _get_embed_fn(self):
        """
        Lazy-initialise the embedding function.

        Priority:
          1. OpenAI via Prodapt gateway (uses base_url override through httpx)
          2. Local SentenceTransformer (no API key needed — always works)

        NOTE: chromadb's built-in OpenAIEmbeddingFunction calls api.openai.com
        directly and does NOT support a custom base_url, so it fails with the
        Prodapt gateway key. We use LangChain OpenAIEmbeddings (which honours
        base_url) wrapped as a custom callable instead.
        """
        if self._embed_fn is None:
            # Try LangChain OpenAI embeddings with Prodapt gateway URL
            if settings.openai_api_key and settings.openai_base_url:
                try:
                    from langchain_openai import OpenAIEmbeddings
                    import urllib3
                    urllib3.disable_warnings()

                    lc_embed = OpenAIEmbeddings(
                        model=settings.openai_embedding_model,
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                        http_client=__import__("httpx").Client(verify=False),
                    )

                    class _LCEmbedFn:
                        """Wraps LangChain embeddings as a ChromaDB EmbeddingFunction."""
                        def __init__(self, lc): self._lc = lc
                        def __call__(self, input):  # noqa: A002
                            return self._lc.embed_documents(input)
                        def name(self) -> str:
                            return "langchain_openai_prodapt"

                    self._embed_fn = _LCEmbedFn(lc_embed)
                    logger.info("ChromaDB embeddings: LangChain OpenAI via Prodapt gateway.")
                except Exception as e:
                    logger.warning(f"LangChain OpenAI embeddings failed ({e}) — using local SentenceTransformer.")
                    self._embed_fn = None

            # Fallback: local SentenceTransformer (no API key, no network needed)
            if self._embed_fn is None:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=settings.local_embedding_model,
                )
                logger.info("ChromaDB embeddings: local SentenceTransformer (all-MiniLM-L6-v2).")

        return self._embed_fn

    def _get_or_create_collection(self, name: str):
        """Get or create a named ChromaDB collection with the embedding function."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )

    # ── Incidents (Pipeline C — primary RAG) ──────────────────────────────────

    def index_incidents(self, force: bool = False) -> int:
        """
        Embed and index all incident records into the 'incidents' collection.

        Args:
            force: If True, delete and re-index the collection.

        Returns:
            Number of documents indexed.
        """
        collection = self._get_or_create_collection(settings.chroma_collection_incidents)

        if not force and collection.count() > 0:
            logger.info(f"Incidents collection already indexed ({collection.count()} docs).")
            return collection.count()

        if force:
            self._client.delete_collection(settings.chroma_collection_incidents)
            collection = self._get_or_create_collection(settings.chroma_collection_incidents)

        df = pd.read_csv(INCIDENTS_CSV)
        logger.info(f"Indexing {len(df)} incidents into ChromaDB...")

        log_pipeline_event("indexing", "ChromaDB Incidents", "start", {"n_docs": len(df)})

        # Batch in groups of 50 to avoid API rate limits
        batch_size = 50
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            collection.add(
                ids=batch["incident_id"].tolist(),
                documents=batch["description"].tolist(),
                metadatas=[
                    {
                        "region":            str(row.region),
                        "equipment_type":    str(row.equipment_type),
                        "outage_event":      str(row.outage_event),
                        "severity":          str(row.severity),
                        "transformer_status": str(row.transformer_status),
                        "timestamp":         str(row.timestamp),
                        "voltage":           float(row.voltage),
                        "grid_frequency":    float(row.grid_frequency),
                    }
                    for row in batch.itertuples()
                ],
            )
            logger.debug(f"Indexed batch {start // batch_size + 1}")

        logger.info(f"Incidents indexed: {collection.count()} documents.")
        log_pipeline_event("indexing", "ChromaDB Incidents", "complete", {"count": collection.count()})
        return collection.count()

    def search_incidents(
        self,
        query: str,
        n_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Semantic search over the incidents collection.

        Args:
            query:     Natural-language search query.
            n_results: Number of candidates to return (before reranking).
            filters:   ChromaDB where-filter dict.
                       e.g. {"region": "Zone_B"} or
                            {"$and": [{"region": "Zone_B"}, {"severity": "critical"}]}

        Returns:
            List of dicts with keys: id, document, metadata, distance.
        """
        collection = self._get_or_create_collection(settings.chroma_collection_incidents)
        if collection.count() == 0:
            logger.warning("Incidents collection is empty — run index_incidents() first.")
            return []

        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": min(n_results, collection.count())}
        if filters:
            kwargs["where"] = filters

        results = collection.query(**kwargs)

        hits = []
        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            hits.append({
                "id":       doc_id,
                "document": doc,
                "metadata": meta,
                "distance": round(dist, 4),
                "score":    round(1 - dist, 4),  # cosine similarity
            })
        return hits

    # ── Grid Stability (Pipeline A — structured text rows) ───────────────────

    def index_stability(self, max_rows: int = 5000, force: bool = False) -> int:
        """
        Serialise DS1 rows as structured text and embed into 'grid_stability' collection.
        We index a representative sample (max_rows) to keep indexing cost manageable.
        """
        collection = self._get_or_create_collection(settings.chroma_collection_stability)

        if not force and collection.count() > 0:
            logger.info(f"Stability collection already indexed ({collection.count()} docs).")
            return collection.count()

        if force:
            self._client.delete_collection(settings.chroma_collection_stability)
            collection = self._get_or_create_collection(settings.chroma_collection_stability)

        df = pd.read_csv(DS1_AUGMENTED).sample(n=min(max_rows, 60000), random_state=42)
        logger.info(f"Indexing {len(df)} stability rows into ChromaDB...")

        def _row_to_text(row) -> str:
            """Serialise a DS1 row to a structured text string for embedding."""
            return (
                f"Region: {row.region} | Equipment: {row.equipment_type} | "
                f"Status: {row.transformer_status} | Outage: {row.outage_event} | "
                f"Stability: {row.stabf} | Stab score: {row.stab:.4f} | "
                f"Frequency: {row.grid_frequency:.2f}Hz | Timestamp: {row.timestamp}"
            )

        batch_size = 100
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            texts = [_row_to_text(r) for r in batch.itertuples()]
            ids   = [f"DS1-{start + i}" for i in range(len(batch))]
            metas = [
                {
                    "region":            str(r.region),
                    "equipment_type":    str(r.equipment_type),
                    "transformer_status": str(r.transformer_status),
                    "outage_event":      str(r.outage_event),
                    "stabf":             str(r.stabf),
                }
                for r in batch.itertuples()
            ]
            collection.add(ids=ids, documents=texts, metadatas=metas)

        logger.info(f"Stability collection indexed: {collection.count()} docs.")
        return collection.count()

    # ── Query Cache ────────────────────────────────────────────────────────────

    def check_cache(self, query: str) -> Optional[str]:
        """
        Check if a semantically similar query was already answered.

        Returns the cached answer string if similarity > threshold, else None.
        """
        collection = self._get_or_create_collection(settings.chroma_collection_cache)
        if collection.count() == 0:
            return None

        results = collection.query(query_texts=[query], n_results=1)
        if not results["ids"][0]:
            return None

        dist = results["distances"][0][0]
        similarity = 1 - dist
        if similarity >= settings.cache_similarity_threshold:
            cached_answer = results["metadatas"][0][0].get("answer", "")
            logger.info(f"Cache hit (similarity={similarity:.3f}) — returning cached response.")
            return cached_answer
        return None

    def store_cache(self, query: str, answer: str) -> None:
        """Store a query + answer in the query cache collection."""
        collection = self._get_or_create_collection(settings.chroma_collection_cache)
        cache_id = hashlib.md5(query.encode()).hexdigest()
        try:
            collection.add(
                ids=[cache_id],
                documents=[query],
                metadatas=[{"answer": answer[:2000]}],  # truncate very long answers
            )
        except Exception:
            # Duplicate ID — update instead
            collection.update(
                ids=[cache_id],
                documents=[query],
                metadatas=[{"answer": answer[:2000]}],
            )


# ── Module-level singleton ─────────────────────────────────────────────────────
_chroma_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Return the shared ChromaStore singleton."""
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store
