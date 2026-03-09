import argparse
import json
import sys
import statistics
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.fallback_index import InMemoryVectorIndex
from app.core.index_client import VectorIndexClient
from app.ml.embeddings import Embedder


def load_queries(query_file: Path) -> List[str]:
    payload = json.loads(query_file.read_text(encoding="utf-8"))
    return payload.get("queries", [])


def make_corpus() -> List[str]:
    return [
        "Knowledge crystals are distilled factual units generated from chunks.",
        "PDF ingestion uses pypdf and source adapters with fallback behavior.",
        "Unavailable connectors include slack, discord, and github with fallback to text.",
        "Vector retrieval mode can run exact baseline or ANN when enabled.",
        "Dashboard tracks p50 p95 latency and retrieval overlap metrics.",
        "FastAPI exposes ingest and query endpoints for Streamlit frontend.",
        "The C++ module is built with pybind11 and optional faiss acceleration.",
        "Ingestion pipeline cleans text, chunks content, and stores crystal summaries."
    ]


def run_queries(
    index_obj,
    embedder: Embedder,
    queries: List[str],
    top_k: int,
) -> Tuple[List[List[str]], List[float]]:
    results: List[List[str]] = []
    latencies_ms: List[float] = []
    for query in queries:
        qv = embedder.encode([query])[0]
        start = time.perf_counter()
        hits = index_obj.search(qv, top_k=top_k)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
        results.append([item[0] for item in hits])
    return results, latencies_ms


def overlap_at_k(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0
    aset = set(a)
    bset = set(b)
    return len(aset.intersection(bset)) / float(len(aset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact baseline vs C++ retrieval path.")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=Path("benchmarks/query_set.json"),
        help="Path to benchmark query set JSON.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = Embedder()
    corpus = make_corpus()
    vectors = embedder.encode(corpus)
    ids = [f"doc-{i}" for i in range(len(corpus))]
    queries = load_queries(args.query_file)

    exact = InMemoryVectorIndex()
    exact.add(ids, vectors)

    candidate = VectorIndexClient()
    candidate.add(ids, vectors)

    exact_results, exact_ms = run_queries(exact, embedder, queries, args.top_k)
    cand_results, cand_ms = run_queries(candidate, embedder, queries, args.top_k)

    overlaps = [overlap_at_k(exact_results[i], cand_results[i]) for i in range(len(queries))]

    print("=== Benchmark Results ===")
    print(f"Queries: {len(queries)} | top_k: {args.top_k}")
    print(f"Candidate mode: {candidate.mode}")
    print(f"Exact latency ms p50: {statistics.median(exact_ms):.3f}")
    print(f"Candidate latency ms p50: {statistics.median(cand_ms):.3f}")
    print(f"Exact latency ms p95: {np.percentile(exact_ms, 95):.3f}")
    print(f"Candidate latency ms p95: {np.percentile(cand_ms, 95):.3f}")
    print(f"Mean top-k overlap vs exact: {statistics.mean(overlaps):.3f}")


if __name__ == "__main__":
    main()
