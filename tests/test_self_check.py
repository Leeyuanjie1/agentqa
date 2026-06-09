from docqa_agent.config import Settings
from docqa_agent.schemas import DocumentChunk, RetrievalCandidate
from docqa_agent.services.self_check import run_self_check


def test_self_check_refuses_when_top_score_is_too_low() -> None:
    settings = Settings(refuse_threshold=0.2, rerank_threshold=0.4)
    candidate = RetrievalCandidate(
        chunk=DocumentChunk(chunk_id="p1-c1", page=1, text="合同总金额为 120000 元。", source_type="paragraph"),
        rerank_score=0.05,
    )

    result = run_self_check("合同总金额为 120000 元。", [candidate], settings)

    assert result.should_refuse is True
    assert result.hallucination_risk == "high"
