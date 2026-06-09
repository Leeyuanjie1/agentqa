from __future__ import annotations

from pathlib import Path

from docqa_agent.logging_utils import get_logger
from docqa_agent.retrieval.text_ops import overlap_score


logger = get_logger(__name__)


class RerankerService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None
        self.backend = "lexical"

    def score(self, question: str, passages: list[str]) -> list[float]:
        if self.model_path.exists():
            try:
                return self._score_with_model(question, passages)
            except Exception as exc:
                logger.warning(
                    "Reranker model at %s failed to load or score. Falling back to lexical overlap. Error: %s",
                    self.model_path,
                    exc,
                )
                self._model = None
                self.backend = "lexical"
        return [overlap_score(question, passage) for passage in passages]

    def _score_with_model(self, question: str, passages: list[str]) -> list[float]:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(str(self.model_path), use_fp16=False)
            self.backend = "flagembedding"
        pairs = [[question, passage] for passage in passages]
        return [float(score) for score in self._model.compute_score(pairs)]
