from __future__ import annotations

from docqa_agent.retrieval.text_ops import tokenize
from docqa_agent.schemas import QueryRouteResult, RetrievalCandidate


TABLE_HINTS = {"表", "表格", "金额", "费用", "数据", "统计", "税率", "付款", "预付款", "验收款", "多少", "账面资金", "账面价值", "账面余额"}
CLAUSE_HINTS = {"条", "条款", "章节", "规定", "约定", "依据", "第"}
HIGH_NO_ANSWER_HINTS = {"税率", "开户行", "账号", "身份证", "邮箱", "电话", "地址", "发票", "折扣"}
TIME_HINTS = {"什么时候", "何时", "截止", "结束", "期限", "到期", "完成"}


def route_question(question: str) -> QueryRouteResult:
    normalized_question = question.lower()
    tokens = set(tokenize(question))
    rewritten_queries: list[str] = []

    if _matches_hint(tokens, normalized_question, HIGH_NO_ANSWER_HINTS):
        rewritten_queries.append(_expand_question(question, tokens))
        return QueryRouteResult(
            route="high_no_answer_risk",
            reason="问题包含高风险字段，文档中常见缺失，优先走保守检索与拒答策略。",
            rewritten_queries=[query for query in rewritten_queries if query != question],
        )

    if _matches_hint(tokens, normalized_question, TABLE_HINTS):
        rewritten_queries.append(_expand_question(question, tokens | {"表格", "金额", "付款"}))
        return QueryRouteResult(
            route="table",
            reason="问题包含金额/表格类信号，优先关注表格证据。",
            rewritten_queries=[query for query in rewritten_queries if query != question],
        )

    if _matches_hint(tokens, normalized_question, CLAUSE_HINTS):
        rewritten_queries.append(_expand_question(question, tokens | {"条款", "规定", "约定"}))
        return QueryRouteResult(
            route="clause",
            reason="问题包含条款编号或规则性表述，优先关注条款段落。",
            rewritten_queries=[query for query in rewritten_queries if query != question],
        )

    if _matches_hint(tokens, normalized_question, TIME_HINTS):
        rewritten_queries.append(_expand_question(question, tokens | {"期限", "结束", "到什么时候"}))

    return QueryRouteResult(
        route="general",
        reason="问题未命中特定表格或条款模式，走通用检索路径。",
        rewritten_queries=[query for query in rewritten_queries if query != question],
    )


def should_retry_retrieval(route: QueryRouteResult, candidates: list[RetrievalCandidate], rerank_threshold: float) -> bool:
    if not route.rewritten_queries:
        return False
    if not candidates:
        return True
    top_score = candidates[0].rerank_score
    if top_score < rerank_threshold:
        return True
    if route.route in {"table", "clause", "high_no_answer_risk"}:
        return True
    return False


def choose_retry_query(route: QueryRouteResult, original_question: str) -> str | None:
    for query in route.rewritten_queries:
        if query.strip() and query.strip() != original_question.strip():
            return query.strip()
    return None


def route_preferred_candidates(route: QueryRouteResult, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    if route.route == "table":
        table_candidates = [candidate for candidate in candidates if candidate.chunk.source_type == "table"]
        return table_candidates or candidates
    if route.route == "clause":
        clause_candidates = [candidate for candidate in candidates if candidate.chunk.clause_id or candidate.chunk.source_type == "clause"]
        return clause_candidates or candidates
    return candidates


def _expand_question(question: str, tokens: set[str]) -> str:
    expanded_tokens = sorted(token for token in tokens if token.strip())
    return question.strip() + " " + " ".join(expanded_tokens)


def _matches_hint(tokens: set[str], normalized_question: str, hints: set[str]) -> bool:
    return any(hint in tokens or hint in normalized_question for hint in hints)
