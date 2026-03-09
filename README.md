# Standalone ECE + LangGraph Service

Production-ready backend that transforms static PDFs into queryable knowledge via an **ETL pipeline** and an **agentic LLM system**. Built for the Cyber Ireland 2022 Report; handles verification, data synthesis, and forecasting with citations and tool traces.

---

## Core deliverables (evaluation)

| Deliverable | Location |
|-------------|----------|
| **ETL pipeline** | `ece_processor.py`, `extraction_service.py`, `chunking_service.py`, `table_extractor.py`, `vector_service.py`, `structured_storage_service.py` — extract, chunk, embed, and load tables + chunks |
| **Agentic backend** | FastAPI app in `app.py`; single **`POST /agent/query`** endpoint for agent orchestration |
| **Execution logs / traces** | `execution_logs/` — JSON logs with query, final answer, and full trace (tool calls, results) for the evaluation test queries. Generate with `python scripts/run_test_queries_and_save_traces.py` after uploading the PDF. |
| **Setup & run** | This README (§8–9) and **QUICKSTART.md** |
| **Architecture justification & limitations** | **ARCHITECTURE_AND_LIMITATIONS.md** — ETL choices, agent framework, toolset; current weaknesses; scaling for production |

### Evaluation scenarios ("moments of truth")

The backend is validated against these three test queries; traces are in `execution_logs/`:

1. **Test 1 – Verification:** *"What is the total number of jobs reported, and where exactly is this stated?"* — Correct number plus exact page and citation.
2. **Test 2 – Data synthesis:** *"Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average."* — Regional tables, comparative metrics, mathematically accurate.
3. **Test 3 – Forecasting:** *"Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?"* — Baseline + target from document; CAGR computed via `calculate_metrics` tool (no LLM math).

---

## 1) What this service does

This module is a self-contained API app that supports two query modes:

1. **Direct retrieval**
   - `POST /search` for semantic vector search
2. **Agentic retrieval (LangGraph)**
   - `POST /agent/query` for tool-routed reasoning over vector + structured data

It processes PDFs through a dual-storage pipeline:

- **Vector DB (Azure Cognitive Search)** for semantic discovery and citations
- **PostgreSQL** for exact table/fact querying

---

## 2) High-level flow

### Ingestion flow (`POST /upload`)
1. **Extract** with Azure Document Intelligence (`extraction_service.py`)
2. **Chunk** text/tables with metadata + bounding boxes (`chunking_service.py`)
3. **Embed + index** chunks in Azure Search (`vector_service.py`)
4. **Extract structured facts/tables** and store in PostgreSQL (`fact_extractor.py`, `table_extractor.py`, `structured_storage_service.py`)

### Query flow (`POST /agent/query`)
1. User question enters LangGraph (`langgraph_agent.py`)
2. Model chooses tools from `tools.py`
3. Tool results are fed back to model until final grounded answer is produced
4. Optional `debug: true` returns full trace (`tool_calls`, `tool_results`, ordered `events`)

### End-to-end lifecycle diagram

```mermaid
flowchart TD
  A[Client] -->|POST /upload| B[FastAPI app.py]
  B --> C[ece_processor.py]
  C --> D[extraction_service.py\nAzure Document Intelligence]
  D --> E[chunking_service.py\nchunk + page metadata + bounding boxes]
  E --> F[vector_service.py\nembeddings + Azure Search index]
  D --> G[fact_extractor.py]
  D --> H[table_extractor.py]
  G --> I[structured_storage_service.py\nPostgreSQL facts]
  H --> J[structured_storage_service.py\nPostgreSQL tables]

  A -->|POST /search| K[Semantic retrieval]
  K --> F

  A -->|POST /agent/query| L[langgraph_agent.py]
  L --> M[LLM tool routing]
  M --> N[tools.py semantic_search]
  M --> O[tools.py query_table / discover_tables]
  M --> P[tools.py calculate_metrics / compare_data]
  N --> F
  O --> I
  O --> J
  P --> L
  F --> L
  I --> L
  J --> L
  L -->|answer + optional trace| A
```

---

## 3) Code structure

```text
.
├── app.py                       # FastAPI app + endpoints
├── langgraph_agent.py           # LangGraph graph + tool-calling loop + trace builder
├── tools.py                     # Async tools used by agent
├── ece_processor.py             # End-to-end ingestion orchestrator
├── extraction_service.py       # Azure Document Intelligence extraction
├── chunking_service.py         # Chunking + metadata + bounding boxes
├── vector_service.py           # Azure Search indexing + semantic retrieval
├── structured_storage_service.py# PostgreSQL storage/query layer
├── fact_extractor.py            # Structured fact extraction
├── table_extractor.py          # Structured table extraction
├── llm_middleware.py           # OpenAI / Azure OpenAI client selection
├── config.py                   # Environment config loader + validation
├── scripts/
│   └── run_test_queries_and_save_traces.py  # Generate execution_logs for evaluation
├── execution_logs/             # Test query traces (query, final_answer, logs)
├── requirements.txt
├── Dockerfile
├── .env.example
├── README.md
├── QUICKSTART.md
└── ARCHITECTURE_AND_LIMITATIONS.md
```

---

## 4) Agent tools available

`tools.py` exposes these tools to LangGraph:

1. `semantic_search`
2. `query_table`
3. `discover_tables`
4. `get_table_info`
5. `calculate_metrics`
6. `get_source_citation`
7. `compare_data`

---

## 5) Storage format

Data is stored in two stores. Shapes below are what the pipeline writes and what the API returns.

**Vector DB (Azure Cognitive Search)**  
Chunks are indexed with: `id`, `title`, `content`, `contentVector`, `file_id`, `file_name`, `namespace`, `page_number` (string form of a list, e.g. `"[1]"`), `bounding_box` (JSON string), `page_info`.  
Search results (`POST /search`, `semantic_search`): `page_number` is returned as a **list of integers**; `bounding_box` as a **dict**. Each hit includes `content`, `file_name`, `page_number`, `bounding_box`, `score`.

**PostgreSQL – Facts table** (`facts`): `id`, `file_id`, `file_name`, `namespace`, `entity_type`, `value`, `page`, `source_quote`, `confidence`, `created_at`. (Not used by agent tools; table kept for optional future use.)

**PostgreSQL – Tables table** (`tables`): `id`, `table_id`, `file_id`, `file_name`, `namespace`, `headers` (JSONB), `data` (JSONB), `metadata` (JSONB), `created_at`. `metadata` includes `table_index`, `has_headers`, and `page_number` (for citations). `get_source_citation` uses `metadata.page_number` for table citations.

---

## 6) API endpoints

### `GET /health`
Service health check.

### `POST /upload`
Upload and process PDF through full ECE pipeline.

### `POST /search`
Semantic search only (direct vector retrieval).

### `POST /agent/query`
LangGraph agent query.

Request:
```json
{
  "query": "Compare cybersecurity jobs trend and give citation",
  "namespace": "cyber-ireland-2022",
  "max_steps": 8,
  "debug": true
}
```

Response (debug mode includes trace):
```json
{
  "status": "success",
  "query": "...",
  "namespace": "cyber-ireland-2022",
  "steps": 3,
  "answer": "...",
  "trace": {
    "events": [],
    "tool_calls": [],
    "tool_results": []
  }
}
```

---

## 7) Environment variables

Copy `.env.example` to `.env` and set required values.

### Required
- `AZURE_AFR_API_KEY`
- `AZURE_AFR_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_KEY`
- `OPENAI_API_KEY` (or Azure OpenAI credentials if vendor switched)
- `POSTGRES_URI`

### Common optional
- `AZURE_INDEX_NAME`
- `EMBEDDING_MODEL`
- `VECTOR_SEARCH_DIMENSIONS`
- `ACTIVE_LLM_VENDOR`
- `CHUNK_SIZE`
- `BUFFER_SIZE`
- `HOST`
- `PORT`

---

## 8) Local run

Use **Python 3.11 or 3.12**. From the project root:

```bash
pip install -r requirements.txt
python app.py
```

Server default: `http://localhost:8001`

---

## 9) Docker deployment

Build:
```bash
docker build -t standalone-ece .
```

Run:
```bash
docker run --env-file .env -p 8001:8001 standalone-ece
```

---

## 10) Useful test calls

### Upload
```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "namespace=cyber-ireland-2022"
```

### Agent query with trace
```bash
curl -X POST "http://localhost:8001/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare cybersecurity jobs trend and give citation",
    "namespace": "cyber-ireland-2022",
    "debug": true
  }'
```

---

## 11) Operational notes

- If `discover_tables` returns 0, structured tables are not yet stored for that namespace.
- Use `debug: true` on agent endpoint to inspect routing/tool behavior.
- Service cleanup is handled on shutdown for vector, postgres, and LLM clients.

- Run `python export_for_evaluation.py` to export all chunks and PostgreSQL tables/facts to `export_evaluation/` for storage and retrieval evaluation.
- Run `python scripts/run_test_queries_and_save_traces.py` (after uploading the PDF) to generate execution logs in `execution_logs/`. Each log contains `query`, `final_answer`, and `logs` (full trace). For **architecture justification** and **limitations** (including scaling for production), see **ARCHITECTURE_AND_LIMITATIONS.md**.
