from __future__ import annotations

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from docqa_agent.config import Settings
from docqa_agent.exceptions import IndexNotFoundError
from docqa_agent.logging_utils import get_logger
from docqa_agent.retrieval.embeddings import EmbeddingService
from docqa_agent.retrieval.reranker import RerankerService
from docqa_agent.retrieval.text_ops import tokenize
from docqa_agent.schemas import DocumentChunk, ParsedDocument, RetrievalCandidate
from docqa_agent.utils import ensure_dir, read_json, write_json, write_jsonl


logger = get_logger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        settings: Settings,
        embedding_backend: str,
        reranker_backend: str,
        embedder: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
    ):
        self.parsed_document = parsed_document
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.settings = settings
        self.embedding_backend = embedding_backend
        self.reranker_backend = reranker_backend
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.embedder = embedder or EmbeddingService(settings.embedding_model_path)
        self.reranker = reranker or RerankerService(settings.reranker_model_path)

    @classmethod
    def build(cls, parsed_document: ParsedDocument, chunks: list[DocumentChunk], settings: Settings) -> "KnowledgeBase":
        embedder = EmbeddingService(settings.embedding_model_path)
        embeddings = embedder.encode([chunk.text for chunk in chunks])
        reranker = RerankerService(settings.reranker_model_path)
        return cls(
            parsed_document=parsed_document,
            chunks=chunks,
            embeddings=embeddings,
            settings=settings,
            embedding_backend=embedder.backend,
            reranker_backend=reranker.backend,
            embedder=embedder,
            reranker=reranker,
        )

    @classmethod
    def load(cls, artifact_dir: Path, settings: Settings) -> "KnowledgeBase":
        required_files = [
            artifact_dir / "parsed_document.json",
            artifact_dir / "chunks.json",
            artifact_dir / "embeddings.npy",
            artifact_dir / "metadata.json",
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        if missing:
            raise IndexNotFoundError(f"Index artifact is incomplete under {artifact_dir}: missing {missing}")
        parsed_document = ParsedDocument.model_validate(read_json(artifact_dir / "parsed_document.json"))
        chunks = [DocumentChunk.model_validate(item) for item in read_json(artifact_dir / "chunks.json")]
        embeddings = np.load(artifact_dir / "embeddings.npy")
        metadata = read_json(artifact_dir / "metadata.json")
        embedder = EmbeddingService(settings.embedding_model_path)
        reranker = RerankerService(settings.reranker_model_path)
        return cls(
            parsed_document=parsed_document,
            chunks=chunks,
            embeddings=embeddings,
            settings=settings,
            embedding_backend=metadata.get("embedding_backend", "unknown"),
            reranker_backend=metadata.get("reranker_backend", "unknown"),
            embedder=embedder,
            reranker=reranker,
        )

    def save(self, artifact_dir: Path) -> None:
        ensure_dir(artifact_dir)
        write_json(artifact_dir / "parsed_document.json", self.parsed_document.model_dump(mode="json"))
        write_json(artifact_dir / "chunks.json", [chunk.model_dump(mode="json") for chunk in self.chunks])
        np.save(artifact_dir / "embeddings.npy", self.embeddings)
        write_json(
            artifact_dir / "metadata.json",
            {
                "embedding_backend": self.embedding_backend,
                "reranker_backend": self.reranker_backend,
                "pdf_type": self.parsed_document.pdf_type,
                "source_pdf": self.parsed_document.pdf_path,
                "source_documents": self.parsed_document.source_documents,
                "total_pages": self.parsed_document.total_pages,
                "total_elements": len(self.parsed_document.elements),
                "total_chunks": len(self.chunks),
            },
        )
        write_jsonl(
            artifact_dir / "chunks.jsonl",
            [chunk.model_dump(mode="json") for chunk in self.chunks],
        )

    def search(self, question: str, top_k: int | None = None) -> list[RetrievalCandidate]:
        top_k = top_k or self.settings.top_k
        if not self.chunks:
            return []

        logger.info("Running retrieval for question=%s", question)
        query_vector = self.embedder.encode([question])[0]
        vector_scores = np.dot(self.embeddings, query_vector)

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(question)), dtype=np.float32)
        bm25_max = float(bm25_scores.max()) if len(bm25_scores) else 0.0
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        vector_top = np.argsort(vector_scores)[::-1][: max(top_k * 3, top_k)]
        bm25_top = np.argsort(bm25_scores)[::-1][: max(top_k * 3, top_k)]
        candidate_indices = list(dict.fromkeys([*vector_top.tolist(), *bm25_top.tolist()]))

        candidates: list[RetrievalCandidate] = []
        for index in candidate_indices:
            blended = (
                float(vector_scores[index]) * self.settings.vector_weight
                + float(bm25_scores[index]) * self.settings.bm25_weight
            )
            candidates.append(
                RetrievalCandidate(
                    chunk=self.chunks[index],
                    vector_score=float(vector_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                    blended_score=blended,
                )
            )

        rerank_scores = self.reranker.score(question, [candidate.chunk.text for candidate in candidates])
        for candidate, score in zip(candidates, rerank_scores, strict=False):
            candidate.rerank_score = float(score)

        candidates.sort(key=lambda item: (item.rerank_score, item.blended_score), reverse=True)
        self.embedding_backend = self.embedder.backend
        self.reranker_backend = self.reranker.backend
        return candidates[:top_k]
