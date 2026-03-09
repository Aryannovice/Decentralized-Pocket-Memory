from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    source_type: str = Field(default="text")
    source_ref: Optional[str] = None
    text: str


class IngestUrlRequest(BaseModel):
    url: str
    source_type: str = Field(default="url")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class IndexModeRequest(BaseModel):
    mode: str = Field(default="flat")


class CrystalOut(BaseModel):
    crystal_id: str
    source_type: str
    source_ref: str
    fact_summary: str
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
