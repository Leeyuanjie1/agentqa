from docqa_agent.schemas import DocumentChunk, QueryRouteResult, RetrievalCandidate
from docqa_agent.services.answer_verifier import verify_answer_support
from docqa_agent.services.query_router import choose_retry_query, route_preferred_candidates, route_question, should_retry_retrieval


def test_route_question_classifies_table_clause_and_high_risk() -> None:
    assert route_question("预付款金额是多少？").route == "table"
    assert route_question("第二条的服务期限是什么？").route == "clause"
    assert route_question("发票税率是多少？").route == "high_no_answer_risk"


def test_retry_and_verification_helpers() -> None:
    route_result = route_question("预付款金额是多少？")
    low_score_candidates = [
        RetrievalCandidate(
            chunk=DocumentChunk(
                chunk_id="doc1-p1-e1-c1",
                source_pdf="finance.pdf",
                page=1,
                text="付款安排见表格，预付款 48000 元。",
                source_type="paragraph",
            ),
            rerank_score=0.05,
        )
    ]

    assert should_retry_retrieval(route_result, low_score_candidates, rerank_threshold=0.15) is True
    assert choose_retry_query(route_result, "预付款金额是多少？") is not None

    verified = verify_answer_support("预付款为 48000 元。", low_score_candidates)
    assert verified.supported is True

    unsupported = verify_answer_support("合同税率为 13%。", low_score_candidates)
    assert unsupported.supported is False

    preferred = route_preferred_candidates(
        QueryRouteResult(route="table", reason="table"),
        [
            RetrievalCandidate(
                chunk=DocumentChunk(
                    chunk_id="doc1-p1-t1",
                    source_pdf="finance.pdf",
                    page=1,
                    text="| 阶段 | 金额 |\n| --- | --- |\n| 预付款 | 48000 |",
                    source_type="table",
                    table_markdown="| 阶段 | 金额 |\n| --- | --- |\n| 预付款 | 48000 |",
                ),
                rerank_score=0.2,
            ),
            low_score_candidates[0],
        ],
    )
    assert preferred[0].chunk.source_type == "table"