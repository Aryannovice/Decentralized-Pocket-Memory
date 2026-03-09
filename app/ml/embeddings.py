import os
from typing import List

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_transformer: bool | None = None,
    ) -> None:
        self.model_name = model_name
        if enable_transformer is None:
            enable_transformer = os.getenv("ENABLE_TRANSFORMER_EMBEDDINGS", "0") == "1"
        self.enable_transformer = enable_transformer
        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self.enable_transformer and self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None
        if self._model is None:
            # Deterministic lightweight fallback if model is unavailable.
            vectors = []
            for text in texts:
                arr = np.zeros(384, dtype=np.float32)
                for idx, ch in enumerate(text[:384]):
                    arr[idx] = (ord(ch) % 67) / 67.0
                vectors.append(arr)
            return np.vstack(vectors)
        return np.asarray(self._model.encode(texts), dtype=np.float32)
