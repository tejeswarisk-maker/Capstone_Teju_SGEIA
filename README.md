# ⚡ SGEIA — Smart Grid Energy Intelligence Assistant

**Prodapt FDE Capstone Project** | Powered by GPT-4o · LangGraph · ChromaDB · XGBoost · DeepEval

---

## Overview

SGEIA is a multi-agent AI system for smart grid operational intelligence. It provides:

- **Natural-language querying** over historical grid incident records
- **Real-time stability assessment** using XGBoost trained on physics telemetry (DS1)
- **Smart meter anomaly detection** from household consumption data (DS2)
- **Root cause analysis** with LLM-synthesised failure explanations
- **Mitigation recommendations** quality-gated by DeepEval faithfulness + LLM-as-judge
- **SHAP explainability** for instability driver identification
- **Streaming REST API** (FastAPI + SSE) + interactive **Streamlit** dashboard

---

## Architecture

```
User Query (Streamlit / API)
        │
        ▼
[Input Validation] ← format + domain keywords + Presidio PII masking
        │
        ▼
[LangGraph StateGraph]
        │
   [Query Router] ──(LLM: gpt-4o-mini)──► routing_decision
        │
   ┌────┴──────────────────────────────────────────┐
   │           │              │           │         │
[Incident]  [Stability]  [SmartMeter]  [Cross    [Out-of-
[Retrieval] [Agent]      [Agent]       Domain]   Scope]
   │            │              │           │
   │    BM25+ChromaDB    XGBoost+IF  (all three)
   │    RRF + Rerank       SHAP       sequential
   │            │
   └────────────┴──────────────────────────────────┐
                                                    │
                                          [Failure Analysis]
                                          (GPT-4o: root cause)
                                                    │
                                       [Recommendation Agent]
                                       (GPT-4o + DeepEval gate
                                        + LLM-as-judge, max 2 retries)
                                                    │
                                         [Synthesise Response]
                                          (string assembly, no LLM)
                                                    │
                                          Final Response (JSON/SSE)
```

### Agent Routing Decisions

| Decision       | Trigger                                   | Agents Invoked                              |
|----------------|-------------------------------------------|---------------------------------------------|
| `incident`     | Zone/equipment/outage specific query      | Retrieval → Failure → Recommendations       |
| `stability`    | Grid health / frequency / stability query | Stability → Failure → Recommendations       |
| `smart_meter`  | Demand / consumption / meter query        | SmartMeter → Recommendations                |
| `cross_domain` | Broad or multi-faceted query              | All three → Failure → Recommendations       |
| `out_of_scope` | Non-grid query                            | Rejection message                           |

---

## Project Structure

```
Capstone_Teju_SGEIA/
├── data/
│   ├── originals/                      # Backup of raw datasets
│   ├── smart_grid_stability_augmented.csv  # DS1 (60K rows, 20 cols)
│   ├── household_power_consumption_augmented.csv  # DS2 (2M rows, 16 cols)
│   └── grid_incidents_synthetic.csv    # Synthetic RAG corpus (200 incidents)
├── frontend/
│   └── app.py                          # Streamlit dashboard
├── logs/
│   ├── sgeia.log                       # Application log (rotating 10MB×5)
│   ├── pipeline_audit.log              # JSON pipeline audit (rotating 20MB×10)
│   └── eval_report_*.json              # DeepEval evaluation reports
├── models/                             # Saved ML model files (.joblib)
├── scripts/
│   └── setup_pipeline.py              # One-time setup script
├── src/
│   ├── config.py                       # Pydantic settings (env vars)
│   ├── logger.py                       # Structured logging setup
│   ├── agents/
│   │   ├── state.py                    # AgentState TypedDict
│   │   ├── graph.py                    # LangGraph StateGraph + run_query()
│   │   ├── query_router.py             # LLM query routing
│   │   ├── retrieval_agent.py          # Hybrid BM25 + ChromaDB + Rerank
│   │   ├── stability_agent.py          # XGBoost stability + SHAP
│   │   ├── smart_meter_agent.py        # DS2 anomaly + demand analysis
│   │   ├── failure_agent.py            # GPT-4o root cause analysis
│   │   └── recommendation_agent.py     # Mitigation + quality gates
│   ├── api/
│   │   └── main.py                     # FastAPI endpoints
│   ├── data/
│   │   ├── augment_ds1.py              # DS1 augmentation
│   │   ├── augment_ds2.py              # DS2 augmentation (chunked)
│   │   └── generate_incidents.py       # Synthetic incident generation
│   ├── evaluation/
│   │   └── deepeval_suite.py           # DeepEval evaluation harness
│   ├── guardrails/
│   │   └── validators.py               # Input validation + PII masking
│   ├── indexing/
│   │   ├── chroma_store.py             # ChromaDB vector store
│   │   └── bm25_index.py               # BM25 keyword index + RRF fusion
│   └── models/
│       ├── model_router.py             # LLM model router with fallbacks
│       ├── stability_model.py          # XGBoost classifier + regressor
│       └── anomaly_model.py            # Isolation Forest anomaly detection
├── .env.example                        # Environment variable template
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd Capstone_Teju_SGEIA
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for Presidio PII masking
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your API keys:
#   OPENAI_API_KEY=sk-...          (required)
#   GOOGLE_API_KEY=AIza...         (optional — Gemini fallback)
#   GROQ_API_KEY=gsk_...           (optional — Llama fallback)
#   LANGCHAIN_API_KEY=ls__...      (optional — LangSmith tracing)
```

### 3. Run One-Time Pipeline Setup

```bash
# Full setup (augment data, train models, index incidents)
python scripts/setup_pipeline.py

# If incident CSV already exists:
python scripts/setup_pipeline.py --skip-incidents

# Force re-run all steps:
python scripts/setup_pipeline.py --force
```

### 4. Start the Backend

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Start the Frontend

```bash
# In a separate terminal:
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## API Reference

Base URL: `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

### POST `/api/query`
Main intelligence endpoint — returns full JSON response.

**Request:**
```json
{
  "query": "Analyse voltage instability in Zone_B transformers",
  "session_id": "optional-uuid-for-continuity",
  "widget_context": null
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "query": "...",
  "routing_decision": "incident",
  "health_score": 72,
  "stability_label": "stable",
  "retrieved_count": 5,
  "root_cause": {
    "probable_cause": "Voltage sag from transformer overload during peak demand",
    "evidence": "Similar incidents INC-0042, INC-0098 show same pattern",
    "confidence": 0.85,
    "related_incident_ids": ["INC-0042", "INC-0098"]
  },
  "mitigation_steps": [
    "1. Immediately reduce load on Zone_B feeder by switching non-critical circuits",
    "..."
  ],
  "judge_score": 4.2,
  "faithfulness_score": 0.89,
  "final_response": "## 🟢 Grid Health: 72/100 ...",
  "processing_ms": 3420
}
```

### POST `/api/chat`
SSE streaming version. Emits events progressively:
```
event: start
event: routing
event: retrieval
event: stability
event: root_cause
event: mitigation
event: done
```

### GET `/api/health`
System health check — verifies ChromaDB, ML models, and data files.

### GET `/api/dashboard/metrics`
Aggregated KPIs: incident counts by severity, stability summary, outage distribution.

### POST `/api/incidents`
Direct incident search with metadata filters.
```json
{
  "query": "transformer failure Zone_C",
  "region": "Zone_C",
  "severity": "critical",
  "n_results": 5
}
```

### POST `/api/stability/check`
Direct model inference (bypasses agent pipeline).
```json
{
  "features": {
    "tau1": 2.5, "tau2": 10.0, "tau3": 3.0, "tau4": 5.0,
    "p1": 2.0, "p2": -1.5, "p3": -2.0, "p4": 1.5,
    "g1": 0.05, "g2": 0.07, "g3": 0.03, "g4": 0.06
  }
}
```

---

## LLM Model Strategy

| Task                     | Primary          | Fallback 1            | Fallback 2         |
|--------------------------|------------------|-----------------------|--------------------|
| Complex (root cause, mitig) | gpt-4o        | Gemini 1.5 Pro        | Llama-3.1-70B      |
| Simple (routing, judge)  | gpt-4o-mini      | Gemini 1.5 Flash      | Llama-3.1-8B       |
| Embeddings               | text-embedding-3-small | (via OpenAI)  | all-MiniLM-L6-v2   |

Fallback is automatic — if an API call fails, `ModelRouter` tries the next tier. Configure in `.env`.

---

## Data Pipelines

### DS1 — Grid Stability (60,000 rows)
- Source: UCI Smart Grid Stability dataset
- Augmented columns: `timestamp`, `region`, `equipment_type`, `transformer_status`, `grid_frequency`, `outage_event`
- Used by: Stability Agent (XGBoost), Anomaly Model (Isolation Forest), ChromaDB stability collection

### DS2 — Smart Meter (2.07M rows)
- Source: UCI Household Power Consumption dataset
- Augmented columns: `timestamp`, `demand_load`, `grid_frequency`, `region`, `equipment_type`, `transformer_status`, `outage_event`
- Used by: Smart Meter Agent (anomaly detection + demand analysis)

### Incident Corpus (200 rows)
- Synthetically generated via GPT-4o-mini
- 8 outage event types across 4 zones and 4 severity levels
- Used by: RAG pipeline (ChromaDB semantic + BM25 keyword search + RRF fusion)

---

## Retrieval Pipeline

```
Query
  │
  ├── BM25 keyword search      → top-20 candidates
  ├── ChromaDB semantic search → top-20 candidates
  │
  ▼
RRF fusion (k_rrf=60) → merged ranked list
  │
  ▼
Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) → top-5 final
```

---

## Quality Gates

Recommendation Agent retries up to 2 times if:
- **DeepEval Faithfulness** < 0.7 (steps not grounded in retrieved incidents)
- **LLM-as-Judge** score < 3.0/5.0 (technical soundness / actionability)

Best attempt is always returned even if gates fail after max retries.

---

## Logging & Monitoring

| Log File            | Content                                      | Rotation       |
|---------------------|----------------------------------------------|----------------|
| `logs/sgeia.log`    | Application logs (INFO+) with request_id     | 10MB × 5 files |
| `logs/pipeline_audit.log` | JSON audit trail per pipeline stage   | 20MB × 10 files |
| `logs/eval_report_*.json` | DeepEval evaluation reports           | Per run        |

**Viewing pipeline flow:**
```bash
# Tail live pipeline audit events:
tail -f logs/pipeline_audit.log | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"

# Filter by request_id:
grep "REQ_ID_HERE" logs/pipeline_audit.log
```

---

## Running Evaluations

```bash
# Full DeepEval suite against live pipeline:
python -m src.evaluation.deepeval_suite

# Dry run (test harness structure only, no API calls):
python -m src.evaluation.deepeval_suite --dry-run

# Single metric:
python -m src.evaluation.deepeval_suite --metric faithfulness

# Specific test cases:
python -m src.evaluation.deepeval_suite --test-ids TC-EVAL-001 TC-EVAL-005

# Results saved to: logs/eval_report_{run_id}.json
```

---

## Example Queries

| Query | Expected Route |
|-------|---------------|
| `Analyse voltage instability in Zone_B transformers` | `incident` |
| `What is the current grid stability and health score?` | `stability` |
| `Detect smart meter consumption anomalies` | `smart_meter` |
| `Find similar incidents to Zone_C partial outage with critical transformer status` | `incident` |
| `Why did the grid frequency deviate last week and what should we do?` | `cross_domain` |
| `Full grid assessment: stability, incidents, and smart meter status` | `cross_domain` |

---

## Tech Stack

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| Agent framework  | LangGraph (StateGraph + MemorySaver checkpoint) |
| LLM              | GPT-4o / GPT-4o-mini (+ Gemini / Groq fallback) |
| Vector store     | ChromaDB (persistent, 4 collections)            |
| Keyword search   | rank-bm25 + RRF fusion                          |
| Reranker         | sentence-transformers cross-encoder (ms-marco)  |
| Stability ML     | XGBoost classifier + regressor                  |
| Anomaly ML       | Isolation Forest (scikit-learn)                 |
| Explainability   | SHAP TreeExplainer                              |
| Evaluation       | DeepEval (Faithfulness, Hallucination, GEval)   |
| PII masking      | Microsoft Presidio                              |
| API              | FastAPI + uvicorn (SSE streaming)               |
| Frontend         | Streamlit + Plotly                              |
| Observability    | LangSmith (tracing) + colorlog (structured logs) |

---

## Environment Variables Reference

| Variable                   | Required | Description                                    |
|----------------------------|----------|------------------------------------------------|
| `OPENAI_API_KEY`           | ✅ Yes   | OpenAI API key (GPT-4o + embeddings)           |
| `GOOGLE_API_KEY`           | Optional | Google AI Studio (Gemini fallback)             |
| `GROQ_API_KEY`             | Optional | Groq (Llama fallback)                          |
| `LANGCHAIN_API_KEY`        | Optional | LangSmith tracing                              |
| `LANGCHAIN_TRACING_V2`     | Optional | `true` to enable LangSmith                     |
| `LANGCHAIN_PROJECT`        | Optional | LangSmith project name                         |
| `LOG_LEVEL`                | Optional | `DEBUG`/`INFO`/`WARNING` (default: `INFO`)     |
| `CHROMA_DB_PATH`           | Optional | ChromaDB persistence dir (default: `./chroma_db`) |
| `MODELS_PATH`              | Optional | ML model save dir (default: `./models`)        |
| `DATA_PATH`                | Optional | Data directory (default: `./data`)             |
| `API_HOST`                 | Optional | FastAPI host (default: `127.0.0.1`)            |
| `API_PORT`                 | Optional | FastAPI port (default: `8000`)                 |

---

## License

Prodapt FDE Capstone Project — Internal Use Only.
