from __future__ import annotations

from docqa_agent.config import Settings
from docqa_agent.retrieval.text_ops import overlap_score, tokenize
from docqa_agent.schemas import AnswerVerificationResult, QueryRouteResult, RetrievalCandidate, SelfCheckResult


def run_self_check(
    answer: str,
    candidates: list[RetrievalCandidate],
    settings: Settings,
    route_result: QueryRouteResult | None = None,
    verification_result: AnswerVerificationResult | None = None,
) -> SelfCheckResult:
    if not candidates:
        return SelfCheckResult(
            grounded=True,
            answerable=False,
            hallucination_risk="low",
            reason="未检索到任何证据，已拒答。",
            should_refuse=True,
        )

    top_score = candidates[0].rerank_score
    citations_text = " ".join(candidate.chunk.text for candidate in candidates[:3])
    support_overlap = overlap_score(answer, citations_text)
    route_penalty = 0.02 if route_result and route_result.route == "high_no_answer_risk" else 0.0
    verified = verification_result.supported if verification_result else support_overlap >= 0.1

    if top_score < settings.refuse_threshold + route_penalty:
        return SelfCheckResult(
            grounded=False,
            answerable=False,
            hallucination_risk="high",
            reason="最高重排分过低，证据不足以支撑回答。",
            should_refuse=True,
        )

    if verification_result and not verification_result.supported and verification_result.support_score < 0.08:
        return SelfCheckResult(
            grounded=False,
            answerable=False,
            hallucination_risk="high",
            reason=f"证据校验失败：{verification_result.reason}",
            should_refuse=True,
        )

    if top_score < settings.rerank_threshold or not verified:
        return SelfCheckResult(
            grounded=False,
            answerable=True,
            hallucination_risk="medium",
            reason="证据存在但答案与证据重合度有限，或 citation 校验不足，需要人工复核。",
            should_refuse=False,
        )

    return SelfCheckResult(
        grounded=True,
        answerable=True,
        hallucination_risk="low",
        reason="答案与高分证据一致。",
        should_refuse=False,
    )
