"""
config.py — Centralised settings for SGEIA.

All environment variables are loaded here via pydantic-settings.
Every other module imports from this file — nothing reads os.environ directly.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application-wide settings loaded from .env (via python-dotenv).
    Field names match the environment variable names exactly.
    """

    # ── OpenAI (primary, paid) ────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"   # override for custom gateways
    openai_model_complex: str = "gpt-4o"          # root-cause, mitigation, synthesis
    openai_model_simple: str = "gpt-4o-mini"      # routing, JSON extraction, judge
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # ── Google AI Studio (free fallback for gpt-4o) ───────────────────────────
    google_api_key: str = ""
    google_model_complex: str = "gemini-1.5-pro"
    google_model_simple: str = "gemini-1.5-flash"

    # ── Groq (free fallback tier 2) ───────────────────────────────────────────
    groq_api_key: str = ""
    groq_model_complex: str = "llama-3.1-70b-versatile"
    groq_model_simple: str = "llama-3.1-8b-instant"

    # ── Fallback embeddings (local, no API) ───────────────────────────────────
    local_embedding_model: str = "all-MiniLM-L6-v2"
    local_embedding_dim: int = 384

    # ── LangSmith tracing ─────────────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "SGEIA-Capstone"

    # ── Paths ─────────────────────────────────────────────────────────────────
    chroma_db_path: str = "./chroma_db"
    models_path: str = "./models_saved"
    data_path: str = "./data"
    log_level: str = "INFO"

    # ── API settings ──────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ── RAG / retrieval settings ───────────────────────────────────────────────
    # Maximum incident chunks returned before reranking
    retrieval_candidate_k: int = 20
    # Final top-K after Cross-Encoder reranking
    retrieval_final_k: int = 5
    # Cosine similarity threshold for query cache hit
    cache_similarity_threshold: float = 0.97
    # DeepEval faithfulness gate — regenerate if below this
    faithfulness_threshold: float = 0.7
    # LLM-as-judge score gate (out of 5)
    judge_score_threshold: float = 3.0

    # ── ChromaDB collection names ──────────────────────────────────────────────
    chroma_collection_incidents: str = "incidents"
    chroma_collection_stability: str = "grid_stability"
    chroma_collection_smart_meter: str = "smart_meter"
    chroma_collection_cache: str = "query_cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance — import this everywhere
settings = Settings()

# ── Expose resolved absolute paths for convenience ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / settings.data_path.lstrip("./")
CHROMA_DIR = PROJECT_ROOT / settings.chroma_db_path.lstrip("./")
MODELS_DIR = PROJECT_ROOT / settings.models_path.lstrip("./")
LOGS_DIR = PROJECT_ROOT / "logs"
SOURCE_FILES_DIR = DATA_DIR / "source_files"
ORIGINALS_DIR = DATA_DIR / "originals"

# Augmented / working file paths
DS1_AUGMENTED = DATA_DIR / "smart_grid_stability_augmented.csv"
DS2_AUGMENTED = DATA_DIR / "household_power_consumption.csv"
INCIDENTS_CSV = DATA_DIR / "grid_incidents_synthetic.csv"

# Original (read-only) source paths
DS1_ORIGINAL = SOURCE_FILES_DIR / "smart_grid_stability_augmented.csv"
DS2_ORIGINAL = SOURCE_FILES_DIR / "household_power_consumption.txt"

# Saved ML model file paths
STABILITY_CLASSIFIER_PATH = MODELS_DIR / "stability_classifier.joblib"
STABILITY_REGRESSOR_PATH = MODELS_DIR / "stability_regressor.joblib"
ANOMALY_DS1_PATH = MODELS_DIR / "anomaly_ds1.joblib"
ANOMALY_DS2_PATH = MODELS_DIR / "anomaly_ds2.joblib"
SCALER_DS1_PATH = MODELS_DIR / "scaler_ds1.joblib"
SCALER_DS2_PATH = MODELS_DIR / "scaler_ds2.joblib"
