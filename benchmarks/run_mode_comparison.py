import argparse
import statistics
import time
from typing import Dict, List

import requests


def run_queries(base_url: str, queries: List[str], top_k: int) -> Dict[str, float]:
    latencies: List[float] = []
    overlaps: List[float] = []
    retrieval_ms: List[float] = []
    for query in queries:
        start = time.perf_counter()
        resp = requests.post(
            f"{base_url}/query",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        overlaps.append(float(data.get("metrics", {}).get("topk_overlap_vs_exact", 0.0)))
        retrieval_ms.append(float(data.get("metrics", {}).get("retrieval", {}).get("total_ms", 0.0)))

    return {
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "mean_overlap_vs_exact": statistics.mean(overlaps) if overlaps else 0.0,
        "mean_retrieval_ms": statistics.mean(retrieval_ms) if retrieval_ms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval modes with the same query set.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    query_set = [
        "What is a knowledge crystal?",
        "How does PDF ingestion work?",
        "What fallback is used when slack source is unavailable?",
        "How is retrieval quality measured?",
        "What does the metrics dashboard track?",
    ]

    for mode in ("flat", "hybrid_binary", "binary_only"):
        mode_resp = requests.post(f"{args.base_url}/index/mode", json={"mode": mode}, timeout=15)
        if mode_resp.status_code >= 400:
            print(f"[{mode}] skipped: {mode_resp.text}")
            continue
        result = run_queries(args.base_url, query_set, args.top_k)
        print(f"mode={mode} p50_ms={result['p50_ms']:.2f} p95_ms={result['p95_ms']:.2f} "
              f"mean_overlap_vs_exact={result['mean_overlap_vs_exact']:.3f} "
              f"mean_retrieval_ms={result['mean_retrieval_ms']:.2f}")


if __name__ == "__main__":
    main()
