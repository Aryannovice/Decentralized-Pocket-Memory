import os
from typing import List, Sequence, Tuple

import numpy as np

from app.core.fallback_index import InMemoryVectorIndex

try:
    # Name expected from pybind11 C++ module build.
    import pocket_memory_cpp  # type: ignore
except Exception:  # pragma: no cover
    pocket_memory_cpp = None


class VectorIndexClient:
    def __init__(
        self,
        mode: str | None = None,
        dim: int = 384,
        hnsw_m: int = 32,
        ef_construction: int = 128,
        ef_search: int = 64,
        ivf_nlist: int = 100,
        ivf_nprobe: int = 10,
    ) -> None:
        mode = mode or os.getenv("PM_INDEX_MODE", "flat")
        self._requested_mode = mode
        self._dim = dim
        self._hnsw_m = hnsw_m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._ivf_nlist = ivf_nlist
        self._ivf_nprobe = ivf_nprobe

        if pocket_memory_cpp is not None:
            self._impl = pocket_memory_cpp.VectorIndex()
            try:
                self._impl.configure(
                    mode=mode,
                    dim=dim,
                    hnsw_m=hnsw_m,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                    ivf_nlist=ivf_nlist,
                    ivf_nprobe=ivf_nprobe,
                )
                self._mode = f"cpp-{self._impl.mode()}"
            except Exception:
                self._impl.configure(
                    mode="flat",
                    dim=dim,
                    hnsw_m=hnsw_m,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                    ivf_nlist=ivf_nlist,
                    ivf_nprobe=ivf_nprobe,
                )
                self._requested_mode = "flat"
                self._mode = "cpp-flat"
        else:
            self._impl = InMemoryVectorIndex()
            self._mode = "python-flat-fallback"

    @property
    def mode(self) -> str:
        return self._mode

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if self._mode.startswith("cpp-"):
            self._impl.add(list(ids), vectors.astype(np.float32))
        else:
            self._impl.add(ids, vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._mode.startswith("cpp-"):
            return self._impl.search(query_vector.astype(np.float32), top_k)
        return self._impl.search(query_vector, top_k)

    def reconfigure(self, mode: str) -> None:
        if not self._mode.startswith("cpp-"):
            self._requested_mode = "flat"
            self._mode = "python-flat-fallback"
            return
        self._requested_mode = mode
        self._impl.configure(
            mode=mode,
            dim=self._dim,
            hnsw_m=self._hnsw_m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            ivf_nlist=self._ivf_nlist,
            ivf_nprobe=self._ivf_nprobe,
        )
        self._mode = f"cpp-{self._impl.mode()}"

    def size(self) -> int:
        if self._mode.startswith("cpp-"):
            return int(self._impl.size())
        return int(self._impl.size())
