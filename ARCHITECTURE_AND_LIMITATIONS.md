# Architecture justification

This document explains the technical choices behind the ETL pipeline, agent framework, and toolset, and outlines current limitations and how to scale for production (as required by the evaluation deliverables).

---

**ETL**
- **Azure Document Intelligence (Form Recognizer)** is used for extraction because it provides layout-aware parsing (paragraphs, tables, bounding regions) and reliable table structure (cells, headers) from PDFs. This avoids brittle regex-only or image-only pipelines and gives us page numbers and coordinates for citations.
- **Chunking** combines token-based text chunking with dedicated table chunks (tables kept intact and converted to HTML for semantic search). Metadata (page_number, bounding_box, file_id) is preserved so the agent can cite sources.
- **Dual storage**: Vector store (Azure Cognitive Search) for semantic retrieval and discovery; PostgreSQL for exact querying of facts and tables. This separates "find relevant context" from "return precise numbers and rows," which is necessary for verification and synthesis questions.

**Agent**
- **LangGraph** is used for orchestration because it gives an explicit graph (LLM to tools to LLM) with state and step limits, making tool routing and traceability straightforward. The model chooses tools and we record every call and result when `debug: true`.
- **Tools** are split by capability: semantic search (vector), query_table / discover_tables (PostgreSQL), calculate_metrics (CAGR and other math), compare_data (regional/temporal comparison), get_source_citation (page and quote). The system prompt instructs the model to use structured tools for numbers and comparisons and to avoid mental math.

**Toolset**
- **Vector vs PostgreSQL**: Vector search is good for "where is this discussed?" and narrative context; PostgreSQL is used for "what is the exact value?" and table filters. Keeping them separate avoids over-relying on the model to parse unstructured chunks for numeric answers.
- **Dedicated CAGR tool**: LLMs are unreliable at compound growth math. `calculate_metrics(metric_type="cagr", ...)` computes CAGR in code so the agent can pass values from lookup/semantic search and return a correct percentage.

---

# Limitations

**Current weaknesses**
- **AFR dependency**: Table and layout quality depend on Azure Document Intelligence. Complex or noisy PDFs can yield missing or mismerged tables; there is no fallback extractor.
- **Single-document focus**: The pipeline and tools are built around one PDF per namespace. Cross-document comparison or "all documents" queries are not first-class.
- **No retries or backoff**: Transient failures (e.g. Azure rate limits, timeouts) are not retried; upload or agent calls can fail without automatic recovery.
- **Fact extraction is heuristic**: Facts are extracted with regex-style patterns (e.g. numbers near "jobs", "percent"). Documents with unusual phrasing may be under-covered.
- **Citation for chunks**: get_source_citation can return table page from metadata and chunk page from search results, but there is no chunk-by-id lookup, so citation is best-effort.

**Scaling for production**
- **Async job queue**: Move PDF ingestion to a background job (e.g. Celery, Redis Queue) so upload returns quickly and processing runs in workers; add webhooks or polling for completion.
- **Caching**: Cache embedding and search results by (query, namespace) with TTL to reduce latency and cost for repeated questions.
- **Multi-tenant namespaces**: Namespaces already isolate data; add per-tenant auth and rate limits so multiple teams or customers can use the same deployment.
- **Monitoring and cost**: Add metrics (e.g. tool call counts, latency p99, embedding tokens) and alerting; track Azure/OpenAI usage and set budgets or per-namespace quotas.
- **Resilience**: Retry with backoff for Azure and PostgreSQL; circuit breakers for external services; optional dead-letter queue for failed ingestions.
