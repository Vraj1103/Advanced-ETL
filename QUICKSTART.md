# Quickstart (Standalone ECE + LangGraph)

## 1) Install

From the project root:

```bash
pip install -r requirements.txt
```

## 2) Configure

```bash
cp .env.example .env
```

Set required values in `.env`:
- `AZURE_AFR_API_KEY`
- `AZURE_AFR_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_KEY`
- `OPENAI_API_KEY`
- `POSTGRES_URI`

## 3) Run API

```bash
python app.py
```

Service starts on `http://localhost:8001`.

## 4) Test pipeline

### Upload PDF
```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@/absolute/path/report.pdf" \
  -F "namespace=cyber-ireland-2022"
```

### Semantic search
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cybersecurity employment trends",
    "namespace": "cyber-ireland-2022",
    "top_k": 5
  }'
```

### Agent query (no trace)
```bash
curl -X POST "http://localhost:8001/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare cybersecurity jobs trend and give citation",
    "namespace": "cyber-ireland-2022"
  }'
```

### Agent query (with trace)
```bash
curl -X POST "http://localhost:8001/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare cybersecurity jobs trend and give citation",
    "namespace": "cyber-ireland-2022",
    "max_steps": 8,
    "debug": true
  }'
```

## 5) Docker run

```bash
docker build -t standalone-ece .
docker run --env-file .env -p 8001:8001 standalone-ece
```

## 6) Debug tips

- If agent says no tables found, upload a document first for that namespace.
- Set `debug: true` to inspect tool routing and intermediate outputs.
- Check `/health` for service status.
