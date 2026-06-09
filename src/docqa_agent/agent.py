from __future__ import annotations

from pathlib import Path

from docqa_agent.config import Settings
from docqa_agent.exceptions import IndexNotFoundError
from docqa_agent.ingestion import parse_documents
from docqa_agent.logging_utils import get_logger
from docqa_agent.retrieval.store import KnowledgeBase
from docqa_agent.schemas import AgentAnswer, AnswerCitation, IngestSummary
from docqa_agent.services.answer_verifier import verify_answer_support
from docqa_agent.services.generator import AnswerGenerator
from docqa_agent.services.query_router import choose_retry_query, route_preferred_candidates, route_question, should_retry_retrieval
from docqa_agent.services.self_check import run_self_check
from docqa_agent.utils import resolve_pdf_inputs


logger = get_logger(__name__)


class DocumentQaAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator = AnswerGenerator(settings)
        self._kb_cache: dict[str, KnowledgeBase] = {}

    def build_index(self, pdf_path: Path | None = None, artifact_dir: Path | None = None) -> KnowledgeBase:
        target_path = pdf_path or self.settings.pdf_path
        pdf_paths = resolve_pdf_inputs(target_path)
        artifact_dir = artifact_dir or self.settings.artifact_dir
        logger.info("Building index for input=%s pdf_count=%s artifact_dir=%s", target_path, len(pdf_paths), artifact_dir)
        parsed_document, chunks = parse_documents(pdf_paths=pdf_paths, settings=self.settings, root_path=target_path)
        kb = KnowledgeBase.build(parsed_document=parsed_document, chunks=chunks, settings=self.settings)
        kb.save(artifact_dir)
        self._kb_cache[str(artifact_dir.resolve())] = kb
        return kb

    def build_index_summary(self, pdf_path: Path | None = None, artifact_dir: Path | None = None) -> IngestSummary:
        kb = self.build_index(pdf_path=pdf_path, artifact_dir=artifact_dir)
        target_dir = artifact_dir or self.settings.artifact_dir
        return IngestSummary(
            pdf_type=kb.parsed_document.pdf_type,
            total_documents=len(kb.parsed_document.source_documents) or 1,
            total_pages=kb.parsed_document.total_pages,
            total_elements=len(kb.parsed_document.elements),
            total_chunks=len(kb.chunks),
            embedding_backend=kb.embedding_backend,
            reranker_backend=kb.reranker_backend,
            artifact_dir=str(target_dir),
        )

    def load_index(self, artifact_dir: Path | None = None) -> KnowledgeBase:
        artifact_dir = artifact_dir or self.settings.artifact_dir
        cache_key = str(artifact_dir.resolve())
        cached = self._kb_cache.get(cache_key)
        if cached is not None:
            return cached
        kb = KnowledgeBase.load(artifact_dir=artifact_dir, settings=self.settings)
        self._kb_cache[cache_key] = kb
        return kb

    def ask(self, question: str, artifact_dir: Path | None = None) -> AgentAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        try:
            kb = self.load_index(artifact_dir=artifact_dir)
        except IndexNotFoundError:
            logger.exception("Index not found for artifact_dir=%s", artifact_dir or self.settings.artifact_dir)
            raise

        route_result = route_question(normalized_question)
        candidates = kb.search(question=normalized_question, top_k=self.settings.top_k)
        retry_used = False
        retry_query = None

        if should_retry_retrieval(route_result, candidates, self.settings.rerank_threshold):
            retry_query = choose_retry_query(route_result, normalized_question)
            if retry_query:
                retry_candidates = kb.search(question=retry_query, top_k=self.settings.top_k)
                retry_used = True
                if _retry_improves_results(retry_candidates, candidates, route_result):
                    candidates = retry_candidates

        routed_candidates = route_preferred_candidates(route_result, candidates)
        answer = self.generator.generate(
            question=normalized_question,
            candidates=routed_candidates,
            route_result=route_result,
        )
        verification_result = verify_answer_support(answer=answer, candidates=routed_candidates)
        self_check = run_self_check(
            answer=answer,
            candidates=routed_candidates,
            settings=self.settings,
            route_result=route_result,
            verification_result=verification_result,
        )

        if self_check.should_refuse:
            answer = "无法根据当前文档证据可靠回答该问题。"

        citations = [
            AnswerCitation(
                source_pdf=candidate.chunk.source_pdf,
                page=candidate.chunk.page,
                chunk_id=candidate.chunk.chunk_id,
                snippet=candidate.chunk.text[:220],
            )
            for candidate in routed_candidates[: self.settings.max_citation_count]
        ]

        return AgentAnswer(
            question=normalized_question,
            answer=answer,
            citations=citations,
            self_check=self_check,
            retrieval=routed_candidates,
            metadata={
                "pdf_type": kb.parsed_document.pdf_type,
                "total_documents": len(kb.parsed_document.source_documents) or 1,
                "embedding_backend": kb.embedding_backend,
                "reranker_backend": kb.reranker_backend,
                "query_route": route_result.model_dump(mode="json"),
                "retrieval_retry": {
                    "used": retry_used,
                    "retry_query": retry_query,
                },
                "answer_verification": verification_result.model_dump(mode="json"),
            },
        )


def _retry_improves_results(
    retry_candidates,
    original_candidates,
    route_result,
) -> bool:
    if not retry_candidates:
        return False
    if not original_candidates:
        return True

    retry_top = retry_candidates[0].rerank_score
    original_top = original_candidates[0].rerank_score
    if retry_top > original_top:
        return True

    if route_result.route == "table":
        retry_has_table = any(candidate.chunk.source_type == "table" for candidate in retry_candidates[:3])
        original_has_table = any(candidate.chunk.source_type == "table" for candidate in original_candidates[:3])
        return retry_has_table and not original_has_table

    if route_result.route == "clause":
        retry_has_clause = any(candidate.chunk.clause_id or candidate.chunk.source_type == "clause" for candidate in retry_candidates[:3])
        original_has_clause = any(
            candidate.chunk.clause_id or candidate.chunk.source_type == "clause" for candidate in original_candidates[:3]
        )
        return retry_has_clause and not original_has_clause

    return False
