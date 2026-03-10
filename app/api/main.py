from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.models import (
    IndexModeRequest,
    IngestGithubRequest,
    IngestTextRequest,
    IngestUrlRequest,
    QueryRequest,
    QueryResponse,
)
from app.services.engine import MemoryEngine

app = FastAPI(title="Decentralized Pocket Memory API", version="0.1.0")
engine = MemoryEngine()
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_mode": engine.index.mode}


@app.get("/index/mode")
def get_index_mode() -> dict:
    return {"mode": engine.index.mode}


@app.post("/index/mode")
def set_index_mode(request: IndexModeRequest) -> dict:
    try:
        return engine.set_index_mode(request.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sources/status")
def source_status() -> dict:
    return engine.sources.status()


@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest) -> dict:
    return engine.ingest_from_source(
        source_type="text",
        payload={
            "text": request.text,
            "source_ref": request.source_ref or "manual",
        },
    )


@app.post("/ingest/url")
def ingest_url(request: IngestUrlRequest) -> dict:
    return engine.ingest_from_source(
        source_type="url",
        payload={"url": request.url, "source_ref": request.url},
    )


@app.post("/ingest/github")
def ingest_github(request: IngestGithubRequest) -> dict:
    return engine.ingest_from_source(
        source_type="github",
        payload={"url": request.url, "source_ref": request.url},
    )


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    source_ref: str = Form(default="uploaded_pdf"),
) -> dict:
    out_path = uploads_dir / file.filename
    data = await file.read()
    out_path.write_bytes(data)
    return engine.ingest_from_source(
        source_type="pdf",
        payload={"file_path": str(out_path), "source_ref": source_ref},
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = engine.query(request.query, top_k=request.top_k)
    return QueryResponse(**result)


@app.get("/metrics")
def metrics() -> dict:
    return engine.get_metrics()
