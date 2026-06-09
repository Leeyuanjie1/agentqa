from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from docqa_agent.logging_utils import get_logger
from docqa_agent.retrieval.text_ops import tokenize


logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_path: str, dimension: int = 512):
        self.model_path = Path(model_path)
        self.dimension = dimension
        self._model = None
        self.backend = "hash"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model_path.exists():
            try:
                return self._encode_transformer(texts)
            except Exception as exc:
                logger.warning(
                    "Embedding model at %s failed to load or encode. Falling back to hash embeddings. Error: %s",
                    self.model_path,
                    exc,
                )
                self._model = None
                self.backend = "hash"
        return self._encode_hash(texts)

    def _encode_transformer(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(self.model_path), trust_remote_code=True)
            self.backend = "sentence-transformers"
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def _encode_hash(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:4], byteorder="big") % self.dimension
                matrix[row_index, slot] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
