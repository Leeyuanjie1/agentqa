from __future__ import annotations

from typing import Iterable

import httpx

from docqa_agent.config import Settings
from docqa_agent.retrieval.text_ops import overlap_score, tokenize
from docqa_agent.schemas import QueryRouteResult, RetrievalCandidate


def _pick_table_rows(question: str, markdown: str, limit: int = 2) -> list[str]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    data_lines = [line for line in lines if not line.startswith("| ---")]
    if len(data_lines) <= 2:
        return data_lines
    header = data_lines[0]
    rows = data_lines[1:]
    ranked_rows = sorted(rows, key=lambda row: overlap_score(question, row), reverse=True)
    return [header, *ranked_rows[:limit]]


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self,
        question: str,
        candidates: list[RetrievalCandidate],
        route_result: QueryRouteResult | None = None,
    ) -> str:
        if self.settings.generator_mode == "openai-compatible":
            return self._generate_with_llm(question, candidates)
        return self._generate_extractive(question, candidates, route_result=route_result)

    def _generate_extractive(
        self,
        question: str,
        candidates: list[RetrievalCandidate],
        route_result: QueryRouteResult | None = None,
    ) -> str:
        if not candidates:
            return "无法从当前文档中找到足够证据回答该问题。"

        preferred_candidates = candidates
        if route_result and route_result.route == "table":
            table_candidates = [candidate for candidate in candidates if candidate.chunk.source_type == "table"]
            preferred_candidates = table_candidates or candidates
        elif route_result and route_result.route == "clause":
            clause_candidates = [candidate for candidate in candidates if candidate.chunk.clause_id or candidate.chunk.source_type == "clause"]
            preferred_candidates = clause_candidates or candidates

        lead = preferred_candidates[0].chunk
        if lead.source_type == "table" and lead.table_markdown:
            picked_rows = _pick_table_rows(question, lead.table_markdown)
            return "根据文档中的表格，最相关的信息如下：\n" + "\n".join(picked_rows)

        passages = []
        for candidate in preferred_candidates[:2]:
            clause_prefix = f"{candidate.chunk.clause_id}：" if candidate.chunk.clause_id else ""
            passages.append(clause_prefix + candidate.chunk.text)
        summary = "；".join(passages)
        return f"根据文档证据，{summary}"

    def _generate_with_llm(self, question: str, candidates: list[RetrievalCandidate]) -> str:
        if not (self.settings.llm_base_url and self.settings.llm_model):
            return self._generate_extractive(question, candidates)

        evidence = "\n\n".join(
            f"[file={candidate.chunk.source_pdf}][page={candidate.chunk.page}][chunk={candidate.chunk.chunk_id}] {candidate.chunk.text}"
            for candidate in candidates[:4]
        )
        prompt = (
            "你是一个只允许基于证据回答的文档问答助手。"
            "如果证据不足，直接回答‘无法根据文档确认’。\n"
            f"问题：{question}\n"
            f"证据：\n{evidence}\n"
            "请用中文输出简洁答案。"
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        response = httpx.post(
            f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.settings.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


def citation_snippets(candidates: Iterable[RetrievalCandidate], limit: int = 3) -> list[str]:
    snippets = []
    for candidate in list(candidates)[:limit]:
        snippets.append(candidate.chunk.text[:220])
    return snippets
