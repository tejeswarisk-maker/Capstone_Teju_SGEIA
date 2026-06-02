# SGEIA — RAG Pipeline Architecture

> **VS Code:** Install **"Mermaid Preview"** extension → open this file → press `Ctrl+Shift+V`

---

## Simple Flow Overview

```
CSV Datasets → Chunking → Embedding → Indexing → ChromaDB
                                                      ↓
User Question → Guardrails → Cache → Orchestrator → Hybrid Search → LLM → DeepEval → Answer
```

---

## Full RAG Pipeline Diagram

```mermaid
flowchart TD

    subgraph OFFLINE["🏗️ OFFLINE — Build Knowledge Base (Run Once)"]
        direction TB
        DS1["📄 DS1 — Grid Stability\n60,000 rows\ntau · p · g features\nSUPPLY SIDE"]
        DS2["📄 DS2 — Smart Meter\n2,000,000 rows\nvoltage · current · power\nDEMAND SIDE"]
        INC["📄 200 Incidents CSV\nGenerated from DS1 + DS2\nThe RAG knowledge base"]
        CHUNK["✂️ CHUNKING\n1 incident = 1 chunk\n40-80 words each"]
        EMBED["🔢 EMBEDDING\nall-MiniLM-L6-v2\n384-dim vector\nRuns on CPU · No internet"]
        BM25["🔍 BM25 INDEX\nKeyword inverted index\nFinds exact: Zone_A · INC-042"]
        CHROMA["🗄️ CHROMADB\nVector store on disk\nCosine similarity search\nMetadata filters supported"]
        DS1 --> INC
        DS2 --> INC
        INC --> CHUNK --> EMBED
        EMBED --> BM25
        EMBED --> CHROMA
    end

    subgraph ONLINE["⚡ ONLINE — Every User Question (Real-time ~2-5s)"]
        direction TB
        USER["👤 USER QUESTION\nPlain English in chat\ne.g. Why is Zone A unstable?"]
        GUARD["🛡️ INPUT GUARDRAILS\n① Format Check\n② Harmful Content Blocked\n③ Domain Relevance Check\n④ PII Masking"]
        CACHE["⚡ CACHE LOOKUP\nSame question before?\ncosine > 0.97 → instant reply!\nElse → run full pipeline"]
        ORCH["🧩 ORCHESTRATOR AGENT\nClassifies intent\nRoutes to right agents only"]
        BM25S["🔍 BM25 SEARCH\nExact keyword matches\nZone · ID · Equipment"]
        SEMS["🧠 SEMANTIC SEARCH\nSimilar meaning\nChromaDB cosine sim"]
        MLS["📊 ML ANALYSIS\nXGBoost → Health Score\nIsolation Forest → Anomaly"]
        RRF["⚗️ RRF FUSION\nCombines BM25 + Semantic\nPicks Top 8 incidents\n+15% better recall"]
        LLM["🤖 LLM — Groq llama\nReads: question + 8 incidents\n+ XGBoost stats + DS2 stats\nWrites grounded answer"]
        DEVAL["🔬 DEEPEVAL QUALITY\n① Faithfulness 0-100%\n② LLM Judge 1-5 · need 3+\n③ Output Safety Pass/Fail\nFail → retry once"]
        ANS["✅ ANSWER\nMarkdown formatted\nDeepEval scores in grey\n👍 👎 feedback buttons"]
        USER --> GUARD
        GUARD -- "PASS" --> CACHE
        GUARD -- "BLOCK" --> ANS
        CACHE -- "HIT instant" --> ANS
        CACHE -- "MISS" --> ORCH
        ORCH --> BM25S & SEMS & MLS
        BM25S --> RRF
        SEMS --> RRF
        MLS --> LLM
        RRF --> LLM
        LLM --> DEVAL
        DEVAL -- "PASS" --> ANS
        DEVAL -- "FAIL retry" --> LLM
    end

    OFFLINE --> USER

    style OFFLINE fill:#EFF6FF,stroke:#065A82,stroke-width:2px
    style ONLINE  fill:#F0FDF4,stroke:#22C55E,stroke-width:2px
    style DS1    fill:#065A82,color:#fff
    style DS2    fill:#065A82,color:#fff
    style INC    fill:#21295C,color:#fff
    style CHUNK  fill:#1C7293,color:#fff
    style EMBED  fill:#1d4ed8,color:#fff
    style BM25   fill:#1C7293,color:#fff
    style CHROMA fill:#065A82,color:#fff
    style USER   fill:#21295C,color:#fff
    style GUARD  fill:#EF4444,color:#fff
    style CACHE  fill:#F59E0B,color:#1a202c
    style ORCH   fill:#21295C,color:#fff
    style BM25S  fill:#1C7293,color:#fff
    style SEMS   fill:#065A82,color:#fff
    style MLS    fill:#0F766E,color:#fff
    style RRF    fill:#7C3AED,color:#fff
    style LLM    fill:#B45309,color:#fff
    style DEVAL  fill:#7C3AED,color:#fff
    style ANS    fill:#22C55E,color:#fff
```

---

## One-Line Summary Per Step

| Step | What | One Line |
|------|------|----------|
| 1 | Load CSV | Read DS1 60K + DS2 2M + Incidents 200 into memory |
| 2 | Chunking | Each incident description = 1 chunk (40-80 words) |
| 3 | Embedding | Convert text to 384-number vector using all-MiniLM-L6-v2 |
| 4 | BM25 Index | Build keyword index for fast exact matching |
| 5 | ChromaDB | Store vectors on disk for cosine similarity search |
| 6 | Guardrails | Block harmful/off-topic, mask PII before pipeline |
| 7 | Cache | Return instant answer if same question seen before |
| 8 | Orchestrator | Decide which agents to run based on question type |
| 9 | BM25 Search | Find incidents with exact zone/ID/equipment keywords |
| 10 | Semantic Search | Find incidents with similar meaning using vectors |
| 11 | ML Analysis | XGBoost gives health score, Isolation Forest flags anomalies |
| 12 | RRF Fusion | Combine BM25 + Semantic and pick best 8 incidents |
| 13 | LLM | Read all context and write grounded answer in plain English |
| 14 | DeepEval | Check faithfulness, judge quality, scan output safety |
| 15 | Answer | Show result with DeepEval scores and feedback buttons |

---

## Key Numbers

| Item | Value |
|------|-------|
| DS1 dataset | 60,000 rows |
| DS2 dataset | 2,000,000 rows |
| Incidents indexed | 200 records |
| Embedding size | 384 dimensions |
| Top-K results | 8 incidents |
| Cache threshold | cosine > 0.97 |
| Judge pass score | >= 3 out of 5 |
| Faithfulness pass | >= 50% |
| LLM response time | ~2 seconds (Groq) |
| Dashboard refresh | every 60 seconds |

---

*SGEIA · Tejeswari S K · Prodapt AFDE Capstone · May 2026*
