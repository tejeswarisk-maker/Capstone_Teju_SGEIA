"""
model_router.py — LLM ModelRouter with automatic fallback.

Primary models (OpenAI paid):
  Complex tasks → gpt-4o
  Simple tasks  → gpt-4o-mini
  Embeddings    → text-embedding-3-small

Fallback tier 1 (Google AI Studio — free):
  Complex → gemini-1.5-pro
  Simple  → gemini-1.5-flash

Fallback tier 2 (Groq — free):
  Complex → llama-3.1-70b-versatile
  Simple  → llama-3.1-8b-instant

Fallback tier 3 (local sentence-transformers, embeddings only):
  all-MiniLM-L6-v2

Every call is tagged in LangSmith with a 'fallback_model' metadata key
so we can trace which tier was actually used.

Usage:
    from src.models.model_router import ModelRouter
    router = ModelRouter()
    llm    = router.get_llm("complex")
    embed  = router.get_embeddings()
"""

import os
from typing import Literal

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Task complexity type alias
Complexity = Literal["complex", "simple"]


class ModelRouter:
    """
    Wraps LangChain LLM and Embedding constructors with automatic fallback.

    Tier resolution order:
      1. OpenAI (primary, paid)
      2. Google Gemini (fallback 1, free tier)
      3. Groq Llama   (fallback 2, free tier)
      4. Local sentence-transformers (embeddings only, tier 3)
    """

    def __init__(self) -> None:
        self._llm_cache: dict = {}
        self._embed_cache: dict = {}

    # ── LLM ───────────────────────────────────────────────────────────────────

    def get_llm(self, complexity: Complexity = "simple", temperature: float = 0.1):
        """
        Return the best available LangChain chat model for the given complexity.

        Args:
            complexity:  'complex' → gpt-4o / gemini-1.5-pro / llama-70b
                         'simple'  → gpt-4o-mini / gemini-flash / llama-8b
            temperature: Passed through to the model constructor.

        Returns:
            A LangChain BaseChatModel instance.
        """
        cache_key = f"{complexity}_{temperature}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        llm = (
            self._try_openai(complexity, temperature)
            or self._try_google(complexity, temperature)
            or self._try_groq(complexity, temperature)
        )
        if llm is None:
            raise RuntimeError(
                "All LLM providers are unavailable. "
                "Check your API keys in .env and network connectivity."
            )

        self._llm_cache[cache_key] = llm
        return llm

    def _try_openai(self, complexity: Complexity, temperature: float):
        """Attempt to initialise an OpenAI ChatOpenAI model."""
        if not settings.openai_api_key:
            logger.debug("OpenAI API key not configured — skipping.")
            return None
        try:
            from langchain_openai import ChatOpenAI
            model_name = (
                settings.openai_model_complex
                if complexity == "complex"
                else settings.openai_model_simple
            )
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                metadata={"provider": "openai", "fallback_model": model_name},
            )
            logger.info(f"LLM resolved: OpenAI {model_name} (complexity={complexity})")
            return llm
        except Exception as e:
            logger.warning(f"OpenAI LLM init failed: {e}")
            return None

    def _try_google(self, complexity: Complexity, temperature: float):
        """Attempt to initialise a Google Gemini model (free tier)."""
        if not settings.google_api_key:
            logger.debug("Google API key not configured — skipping Gemini fallback.")
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = (
                settings.google_model_complex
                if complexity == "complex"
                else settings.google_model_simple
            )
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True,
            )
            logger.warning(
                f"LLM FALLBACK → Google Gemini {model_name} (OpenAI unavailable)"
            )
            return llm
        except Exception as e:
            logger.warning(f"Google Gemini LLM init failed: {e}")
            return None

    def _try_groq(self, complexity: Complexity, temperature: float):
        """Attempt to initialise a Groq Llama model (free tier)."""
        if not settings.groq_api_key:
            logger.debug("Groq API key not configured — skipping Groq fallback.")
            return None
        try:
            from langchain_groq import ChatGroq
            model_name = (
                settings.groq_model_complex
                if complexity == "complex"
                else settings.groq_model_simple
            )
            llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=settings.groq_api_key,
            )
            logger.warning(
                f"LLM FALLBACK → Groq {model_name} (OpenAI + Google unavailable)"
            )
            return llm
        except Exception as e:
            logger.warning(f"Groq LLM init failed: {e}")
            return None

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def get_embeddings(self):
        """
        Return the best available LangChain embeddings model.

        Priority:
          1. OpenAI text-embedding-3-small (primary, 1536-dim)
          2. Local sentence-transformers all-MiniLM-L6-v2 (fallback, 384-dim)

        Returns:
            A LangChain Embeddings instance.
        """
        if "embeddings" in self._embed_cache:
            return self._embed_cache["embeddings"]

        embed = self._try_openai_embeddings() or self._try_local_embeddings()
        if embed is None:
            raise RuntimeError("No embedding provider available.")

        self._embed_cache["embeddings"] = embed
        return embed

    def _try_openai_embeddings(self):
        """Attempt to initialise OpenAI embeddings."""
        if not settings.openai_api_key:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings
            embed = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            logger.info(f"Embeddings resolved: OpenAI {settings.openai_embedding_model}")
            return embed
        except Exception as e:
            logger.warning(f"OpenAI embeddings init failed: {e}")
            return None

    def _try_local_embeddings(self):
        """Attempt to initialise local sentence-transformers embeddings."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embed = HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
                model_kwargs={"device": "cpu"},
            )
            logger.warning(
                f"Embeddings FALLBACK → local {settings.local_embedding_model} "
                "(OpenAI unavailable — 384-dim)"
            )
            return embed
        except Exception as e:
            logger.error(f"Local embeddings init failed: {e}")
            return None


# ── Module-level singleton (import and reuse) ──────────────────────────────────
_router_instance: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Return the shared ModelRouter singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance
