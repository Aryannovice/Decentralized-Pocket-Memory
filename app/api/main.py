from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.models import (
    CrystalRegistryListResponse,
    CrystalRegistryRecord,
    IndexModeRequest,
    IngestGithubRequest,
    IngestRedditRequest,
    QueryUsageListResponse,
    QueryUsageRecord,
    StatePathRequest,
    TransferListResponse,
    TransferOut,
    TransferRequest,
    IngestTextRequest,
    IngestUrlRequest,
    QueryRequest,
    QueryResponse,
    WalletCrystalOut,
    WalletOut,
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
            "creator_id": request.creator_id or "local_user",
        },
    )


@app.post("/ingest/url")
def ingest_url(request: IngestUrlRequest) -> dict:
    return engine.ingest_from_source(
        source_type="url",
        payload={"url": request.url, "source_ref": request.url, "creator_id": request.creator_id or "local_user"},
    )


@app.post("/ingest/github")
def ingest_github(request: IngestGithubRequest) -> dict:
    return engine.ingest_from_source(
        source_type="github",
        payload={"url": request.url, "source_ref": request.url, "creator_id": request.creator_id or "local_user"},
    )


@app.post("/ingest/reddit")
def ingest_reddit(request: IngestRedditRequest) -> dict:
    return engine.ingest_from_source(
        source_type="reddit",
        payload={"url": request.url, "source_ref": request.url, "creator_id": request.creator_id or "local_user"},
    )


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    source_ref: str = Form(default="uploaded_pdf"),
    creator_id: str = Form(default="local_user"),
) -> dict:
    out_path = uploads_dir / file.filename
    data = await file.read()
    out_path.write_bytes(data)
    return engine.ingest_from_source(
        source_type="pdf",
        payload={"file_path": str(out_path), "source_ref": source_ref, "creator_id": creator_id},
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = engine.query(request.query, top_k=request.top_k, source_types=request.source_types)
    return QueryResponse(**result)


@app.get("/metrics")
def metrics() -> dict:
    return engine.get_metrics()


@app.post("/state/save")
def save_state(request: StatePathRequest) -> dict:
    try:
        return engine.save_state(request.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/state/load")
def load_state(request: StatePathRequest) -> dict:
    try:
        return engine.load_state(request.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/registry/crystals", response_model=CrystalRegistryListResponse)
def list_registry_crystals(limit: int = 200) -> CrystalRegistryListResponse:
    items = engine.list_crystal_registry(limit=limit)
    return CrystalRegistryListResponse(items=items, count=len(items))


@app.get("/registry/crystals/{crystal_id}", response_model=CrystalRegistryRecord)
def get_registry_crystal(crystal_id: str) -> CrystalRegistryRecord:
    item = engine.get_crystal_registry(crystal_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Crystal not found: {crystal_id}")
    return CrystalRegistryRecord(**item)


@app.get("/registry/queries", response_model=QueryUsageListResponse)
def list_registry_queries(limit: int = 200) -> QueryUsageListResponse:
    items = engine.list_query_usage(limit=limit)
    return QueryUsageListResponse(items=items, count=len(items))


@app.get("/registry/queries/{query_id}", response_model=QueryUsageRecord)
def get_registry_query(query_id: str) -> QueryUsageRecord:
    item = engine.get_query_usage(query_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Query not found: {query_id}")
    return QueryUsageRecord(**item)


@app.get("/registry/leaderboard", response_model=CrystalRegistryListResponse)
def registry_leaderboard(limit: int = 50) -> CrystalRegistryListResponse:
    items = engine.get_registry_leaderboard(limit=limit)
    return CrystalRegistryListResponse(items=items, count=len(items))


@app.get("/wallets/{owner_id}", response_model=WalletOut)
def get_wallet(owner_id: str) -> WalletOut:
    return WalletOut(**engine.get_wallet_snapshot(owner_id))


@app.get("/wallets/{owner_id}/crystals", response_model=list[WalletCrystalOut])
def get_wallet_crystals(owner_id: str, limit: int = 200) -> list[WalletCrystalOut]:
    items = engine.list_wallet_crystals(owner_id=owner_id, limit=limit)
    return [WalletCrystalOut(**item) for item in items]


@app.get("/wallets/{owner_id}/transfers", response_model=TransferListResponse)
def get_wallet_transfers(owner_id: str, limit: int = 200) -> TransferListResponse:
    items = engine.list_wallet_transfers(owner_id=owner_id, limit=limit)
    return TransferListResponse(items=[TransferOut(**item) for item in items], count=len(items))


@app.post("/wallets/transfer")
def transfer_wallet_crystal(request: TransferRequest) -> dict:
    try:
        return engine.transfer_crystal(
            crystal_id=request.crystal_id,
            new_owner_id=request.new_owner_id,
            actor_id=request.actor_id,
            reason=request.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ===== BLOCKCHAIN INTEGRATION ENDPOINTS =====

@app.get("/blockchain/status")
def get_blockchain_status() -> dict:
    """Get blockchain integration status and account info."""
    return engine.get_blockchain_status()


@app.get("/blockchain/account")
def get_blockchain_account() -> dict:
    """Get blockchain account information."""
    return engine.get_blockchain_account_info()


@app.post("/blockchain/verify/{crystal_id}")
def verify_crystal_blockchain(crystal_id: str) -> dict:
    """Verify a crystal's proof hash on the blockchain."""
    try:
        result = engine.verify_crystal_on_blockchain(crystal_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
