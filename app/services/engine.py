import time
import uuid
import zlib
from typing import Dict, List

import numpy as np

from app.core.index_client import VectorIndexClient
from app.ml.chunking import chunk_text
from app.ml.distill import distill_chunk
from app.ml.embeddings import Embedder
from app.services.source_registry import SourceRegistry
from app.services.telemetry import Telemetry


class MemoryEngine:
    def __init__(self) -> None:
        self.sources = SourceRegistry()
        self.embedder = Embedder()
        self.index = VectorIndexClient()
        self.crystals: Dict[str, dict] = {}
        self._vectors: List[np.ndarray] = []
        self._vector_ids: List[str] = []
        self.telemetry = Telemetry()

    def _memory_stats(self) -> tuple[int, int]:
        crystal_bytes = sum(len(c["fact_summary"].encode("utf-8")) for c in self.crystals.values())
        compressed_bytes = sum(
            len(zlib.compress(c["fact_summary"].encode("utf-8"))) for c in self.crystals.values()
        )
        return crystal_bytes, compressed_bytes

    def _exact_overlap(self, query_vector: np.ndarray, approx_ids: List[str], top_k: int) -> float:
        if not self._vectors or not approx_ids:
            return 0.0
        vecs = np.vstack(self._vectors)
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) + 1e-8
        v_norm = np.linalg.norm(vecs, axis=1) + 1e-8
        scores = (vecs @ q) / (v_norm * q_norm)
        idx = np.argsort(-scores)[:top_k]
        exact_ids = [self._vector_ids[i] for i in idx]
        return len(set(exact_ids).intersection(set(approx_ids))) / float(len(exact_ids) or 1)

    def get_metrics(self) -> dict:
        crystal_bytes, compressed_bytes = self._memory_stats()
        return self.telemetry.snapshot(
            crystal_count=len(self.crystals),
            crystal_bytes=crystal_bytes,
            compressed_bytes=compressed_bytes,
            index_size=self.index.size(),
        )

    def set_index_mode(self, mode: str) -> dict:
        self.index.reconfigure(mode)
        if self._vectors and self._vector_ids:
            matrix = np.vstack(self._vectors).astype(np.float32)
            self.index.add(self._vector_ids, matrix)
        return {"mode": self.index.mode, "index_size": self.index.size()}

    def ingest_text(self, text: str, source_type: str = "text", source_ref: str = "manual") -> dict:
        start = time.perf_counter()
        chunks = chunk_text(text)
        if not chunks:
            return {"ingested": 0, "message": "No usable text found."}

        distilled = [distill_chunk(chunk) for chunk in chunks]
        summaries = [item["fact_summary"] for item in distilled]
        vectors = self.embedder.encode(summaries)

        ids: List[str] = []
        for idx, summary in enumerate(summaries):
            crystal_id = str(uuid.uuid4())
            ids.append(crystal_id)
            self._vector_ids.append(crystal_id)
            self._vectors.append(vectors[idx])
            self.crystals[crystal_id] = {
                "crystal_id": crystal_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "chunk_index": idx,
                "fact_summary": summary,
                "metadata": {},
            }

        self.index.add(ids, vectors)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.record_ingest(
            chunks=len(ids),
            elapsed_ms=elapsed_ms,
            source_type=source_type,
            used_fallback=False,
        )
        return {"ingested": len(ids), "index_mode": self.index.mode, "elapsed_ms": elapsed_ms}

    def ingest_from_source(self, source_type: str, payload: dict) -> dict:
        if source_type == "text":
            text = payload.get("text", "")
            return self.ingest_text(text=text, source_type="text", source_ref=payload.get("source_ref", "manual"))

        adapter = self.sources.get(source_type)
        enabled, message = adapter.status()
        if not enabled:
            self.telemetry.record_ingest(
                chunks=0,
                elapsed_ms=0.0,
                source_type=source_type,
                used_fallback=True,
            )
            return {
                "ingested": 0,
                "message": f"{source_type} unavailable: {message}",
                "fallback": "Use pdf/url/text.",
            }

        text = adapter.read(payload)
        source_ref = payload.get("source_ref") or payload.get("url") or payload.get("file_path") or source_type
        return self.ingest_text(text=text, source_type=source_type, source_ref=source_ref)

    def query(self, query: str, top_k: int = 5) -> dict:
        start = time.perf_counter()
        qv = self.embedder.encode([query])[0]
        hits = self.index.search(qv, top_k=top_k)
        crystals = []
        for crystal_id, score in hits:
            crystal = self.crystals.get(crystal_id)
            if crystal is None:
                continue
            item = dict(crystal)
            item["score"] = score
            crystals.append(item)

        if not crystals:
            answer = "No relevant memory crystals found yet. Ingest documents first."
        else:
            answer_parts = [c["fact_summary"] for c in crystals[:3]]
            answer = " ".join(answer_parts)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        approx_ids = [c["crystal_id"] for c in crystals]
        overlap = self._exact_overlap(qv, approx_ids, top_k)
        score_mean = float(np.mean([c["score"] for c in crystals])) if crystals else 0.0
        citation_precision_proxy = 1.0 if crystals else 0.0
        self.telemetry.record_query(
            latency_ms=elapsed_ms,
            overlap=overlap,
            score_mean=score_mean,
            citation_precision_proxy=citation_precision_proxy,
        )
        return {
            "answer": answer,
            "crystals": crystals,
            "metrics": {
                "query_latency_ms": elapsed_ms,
                "topk_overlap_vs_exact": overlap,
            },
        }
