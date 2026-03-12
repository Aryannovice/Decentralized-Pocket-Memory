from typing import List, Sequence, Tuple

import numpy as np


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vectors: np.ndarray = np.zeros((0, 384), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        if self._vectors.size == 0:
            self._vectors = vectors.astype(np.float32)
        else:
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._ids.extend(list(ids))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._vectors.size == 0:
            return []
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) + 1e-8
        v_norm = np.linalg.norm(self._vectors, axis=1) + 1e-8
        scores = (self._vectors @ q) / (v_norm * q_norm)
        idx = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def search_with_stats(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[List[Tuple[str, float]], dict]:
        results = self.search(query_vector, top_k=top_k)
        stats = {
            "prefilter_candidates": len(self._ids),
            "prefilter_ms": 0.0,
            "rerank_ms": 0.0,
            "total_ms": 0.0,
        }
        return results, stats

    def size(self) -> int:
        return len(self._ids)

    def save_state(self, path: str) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                ids=np.asarray(self._ids, dtype=object),
                vectors=self._vectors.astype(np.float32),
            )

    def load_state(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self._ids = [str(x) for x in data["ids"].tolist()]
        self._vectors = np.asarray(data["vectors"], dtype=np.float32)
