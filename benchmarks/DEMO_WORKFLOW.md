# Demo Workflow: Beyond Plain RAG

Use this workflow to demonstrate that the MVP is more than basic RAG.

## 1) Start services

```bash
uvicorn app.api.main:app --reload
```

```bash
streamlit run streamlit_app.py
```

## 2) Ingest a sample document

- Upload a PDF (or use URL/text ingestion).
- Confirm `ingested > 0`.

## 3) Run baseline benchmark

```bash
python benchmarks/run_benchmark.py
```

Capture:
- p50/p95 latency
- mean top-k overlap vs exact
- candidate mode (`python-flat-fallback`, `cpp-flat`, etc.)

## 4) Compare retrieval modes

```bash
python benchmarks/run_mode_comparison.py --base-url http://127.0.0.1:8000
```

Capture:
- `flat` vs `hnsw` vs `ivfpq` latency
- overlap-vs-exact proxy

## 5) Show dashboard evidence

In Streamlit:
- switch retrieval mode
- run same query set
- refresh metrics dashboard

Show:
- latency p50/p95 changes
- overlap-vs-exact trend
- citation precision proxy
- crystal count and compression ratio
- source fallback counts

## 6) Final demo claim

Positioning statement:
- This is not just chunk retrieval + generation.
- The system uses **knowledge crystals**, **mode-configurable C++ retrieval**, and **observable quality/performance metrics** with source reliability tracking.
