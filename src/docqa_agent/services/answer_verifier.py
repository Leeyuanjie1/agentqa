from __future__ import annotations

from docqa_agent.retrieval.text_ops import overlap_score
from docqa_agent.schemas import AnswerVerificationResult, RetrievalCandidate


def verify_answer_support(answer: str, candidates: list[RetrievalCandidate]) -> AnswerVerificationResult:
    if not candidates:
        return AnswerVerificationResult(
            supported=False,
            support_score=0.0,
            matched_citations=0,
            reason="没有可用 citation，无法验证答案支撑度。",
        )

    citation_scores = [overlap_score(answer, candidate.chunk.text) for candidate in candidates[:3]]
    matched_citations = sum(1 for score in citation_scores if score >= 0.12)
    support_score = max(citation_scores) if citation_scores else 0.0

    if support_score >= 0.18 or matched_citations >= 2:
        return AnswerVerificationResult(
            supported=True,
            support_score=support_score,
            matched_citations=matched_citations,
            reason="答案与 citation 片段存在明确重合，可认为有证据支撑。",
        )

    return AnswerVerificationResult(
        supported=False,
        support_score=support_score,
        matched_citations=matched_citations,
        reason="答案与 citation 重合度偏低，支撑不足。",
    )
