from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    source_type: str = Field(default="text")
    source_ref: Optional[str] = None
    creator_id: Optional[str] = None
    text: str


class IngestUrlRequest(BaseModel):
    url: str
    source_type: str = Field(default="url")
    creator_id: Optional[str] = None


class IngestGithubRequest(BaseModel):
    url: str
    source_type: str = Field(default="github")
    creator_id: Optional[str] = None


class IngestRedditRequest(BaseModel):
    url: str
    source_type: str = Field(default="reddit")
    creator_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source_types: Optional[List[str]] = None


class IndexModeRequest(BaseModel):
    mode: str = Field(default="flat")


class StatePathRequest(BaseModel):
    path: str = Field(default="data/state")


class CrystalOut(BaseModel):
    crystal_id: str
    source_type: str
    source_ref: str
    fact_summary: str
    clean_summary: Optional[str] = None
    preview_summary: Optional[str] = None
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    crystals: List[CrystalOut]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SourceStatus(BaseModel):
    source: str
    enabled: bool
    message: str


class CrystalRegistryRecord(BaseModel):
    crystal_id: str
    creator_id: str
    owner_id: str = ""
    source_url: str
    content_hash: str
    embedding_hash: str
    crystal_proof_hash: str = ""
    created_at: str
    usage_count: int = 0
    reward_balance: float = 0.0
    contribution_total: float = 0.0
    last_contribution: float = 0.0
    last_reward_delta: float = 0.0


class CrystalUsageRef(BaseModel):
    crystal_id: str
    similarity: float = 0.0
    rank: int = 0
    contribution_score: float = 0.0
    reward_delta: float = 0.0


class QueryUsageRecord(BaseModel):
    query_id: str
    query_text: str
    created_at: str
    latency_ms: float
    reward_pool: float = 0.0
    crystals_used: List[CrystalUsageRef] = Field(default_factory=list)


class CrystalRegistryListResponse(BaseModel):
    items: List[CrystalRegistryRecord] = Field(default_factory=list)
    count: int = 0


class QueryUsageListResponse(BaseModel):
    items: List[QueryUsageRecord] = Field(default_factory=list)
    count: int = 0


class WalletOut(BaseModel):
    wallet_id: str
    owner_id: str
    created_at: str
    crystal_ids: List[str] = Field(default_factory=list)
    crystal_count: int = 0
    balance: float = 0.0


class WalletCrystalOut(BaseModel):
    crystal_id: str
    creator_id: str
    owner_id: str
    source_url: str
    content_hash: str = ""
    embedding_hash: str = ""
    crystal_proof_hash: str = ""
    created_at: str
    usage_count: int = 0
    reward_balance: float = 0.0
    contribution_total: float = 0.0


class TransferRequest(BaseModel):
    crystal_id: str
    new_owner_id: str
    actor_id: Optional[str] = None
    reason: Optional[str] = None


class TransferOut(BaseModel):
    transfer_id: str
    crystal_id: str
    from_owner_id: str
    to_owner_id: str
    transferred_at: str
    reason: str = ""


class TransferListResponse(BaseModel):
    items: List[TransferOut] = Field(default_factory=list)
    count: int = 0
