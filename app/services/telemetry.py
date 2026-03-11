import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from time import time
from typing import Deque, Dict, List

import numpy as np


@dataclass
class IngestEvent:
    chunks: int
    elapsed_ms: float


class Telemetry:
    def __init__(self, max_points: int = 1000) -> None:
        self.max_points = max_points
        self.query_latencies_ms: Deque[float] = deque(maxlen=max_points)
        self.ingest_events: Deque[IngestEvent] = deque(maxlen=max_points)
        self.retrieval_overlap: Deque[float] = deque(maxlen=max_points)
        self.query_scores_mean: Deque[float] = deque(maxlen=max_points)
        self.citation_precision_proxy: Deque[float] = deque(maxlen=max_points)
        self.prefilter_candidates: Deque[float] = deque(maxlen=max_points)
        self.prefilter_ms: Deque[float] = deque(maxlen=max_points)
        self.rerank_ms: Deque[float] = deque(maxlen=max_points)
        self.total_retrieval_ms: Deque[float] = deque(maxlen=max_points)
        self.source_counts: Dict[str, int] = defaultdict(int)
        self.fallback_counts: Dict[str, int] = defaultdict(int)
        self.started_at = time()

    def record_ingest(self, chunks: int, elapsed_ms: float, source_type: str, used_fallback: bool) -> None:
        self.ingest_events.append(IngestEvent(chunks=chunks, elapsed_ms=elapsed_ms))
        self.source_counts[source_type] += 1
        if used_fallback:
            self.fallback_counts[source_type] += 1

    def record_query(
        self,
        latency_ms: float,
        overlap: float,
        score_mean: float,
        citation_precision_proxy: float,
        prefilter_candidates: float = 0.0,
        prefilter_ms: float = 0.0,
        rerank_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> None:
        self.query_latencies_ms.append(latency_ms)
        self.retrieval_overlap.append(overlap)
        self.query_scores_mean.append(score_mean)
        self.citation_precision_proxy.append(citation_precision_proxy)
        self.prefilter_candidates.append(prefilter_candidates)
        self.prefilter_ms.append(prefilter_ms)
        self.rerank_ms.append(rerank_ms)
        self.total_retrieval_ms.append(total_ms)

    def _percentile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float32), q))

    def snapshot(self, crystal_count: int, crystal_bytes: int, compressed_bytes: int, index_size: int) -> dict:
        q_lat = list(self.query_latencies_ms)
        ingest = list(self.ingest_events)
        overlaps = list(self.retrieval_overlap)
        score_mean = list(self.query_scores_mean)
        cites = list(self.citation_precision_proxy)
        pre_candidates = list(self.prefilter_candidates)
        pre_ms = list(self.prefilter_ms)
        rerank_ms = list(self.rerank_ms)
        total_ms = list(self.total_retrieval_ms)

        total_chunks = sum(e.chunks for e in ingest)
        total_ingest_ms = sum(e.elapsed_ms for e in ingest)
        ingest_chunks_per_sec = (1000.0 * total_chunks / total_ingest_ms) if total_ingest_ms > 0 else 0.0
        uptime_sec = max(time() - self.started_at, 1e-6)
        qps = len(q_lat) / uptime_sec
        compression_ratio = (compressed_bytes / crystal_bytes) if crystal_bytes > 0 else 1.0

        return {
            "query_latency_ms": {
                "p50": self._percentile(q_lat, 50),
                "p95": self._percentile(q_lat, 95),
                "max": max(q_lat) if q_lat else 0.0,
            },
            "throughput": {
                "queries_per_sec": qps,
                "ingest_chunks_per_sec": ingest_chunks_per_sec,
            },
            "retrieval": {
                "mean_topk_overlap_vs_exact": statistics.mean(overlaps) if overlaps else 0.0,
                "mean_score": statistics.mean(score_mean) if score_mean else 0.0,
                "mean_prefilter_candidates": statistics.mean(pre_candidates) if pre_candidates else 0.0,
                "mean_prefilter_ms": statistics.mean(pre_ms) if pre_ms else 0.0,
                "mean_rerank_ms": statistics.mean(rerank_ms) if rerank_ms else 0.0,
                "mean_total_retrieval_ms": statistics.mean(total_ms) if total_ms else 0.0,
            },
            "citation": {
                "precision_proxy": statistics.mean(cites) if cites else 0.0,
            },
            "memory": {
                "crystal_count": crystal_count,
                "index_size": index_size,
                "crystal_bytes": crystal_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compression_ratio,
            },
            "sources": {
                "source_counts": dict(self.source_counts),
                "fallback_counts": dict(self.fallback_counts),
            },
            "samples": {
                "queries": len(q_lat),
                "ingests": len(ingest),
            },
            "series": {
                "recent_query_latencies_ms": q_lat[-50:],
                "recent_overlap_vs_exact": overlaps[-50:],
            },
        }
