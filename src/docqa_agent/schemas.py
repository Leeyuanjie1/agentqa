from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PdfType = Literal["text", "scanned", "mixed"]
ElementType = Literal["paragraph", "clause", "table"]
QuestionRoute = Literal["general", "clause", "table", "high_no_answer_risk"]


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""


class DocumentElement(BaseModel):
    page: int
    element_type: ElementType
    text: str
    clause_id: str | None = None
    table: TableData | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    source_pdf: str
    page: int
    text: str
    source_type: ElementType
    clause_id: str | None = None
    table_markdown: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    pdf_path: str
    pdf_type: PdfType
    total_pages: int
    elements: list[DocumentElement]
    source_documents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    chunk: DocumentChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    blended_score: float = 0.0
    rerank_score: float = 0.0


class AnswerCitation(BaseModel):
    source_pdf: str
    page: int
    chunk_id: str
    snippet: str


class SelfCheckResult(BaseModel):
    grounded: bool
    answerable: bool
    hallucination_risk: Literal["low", "medium", "high"]
    reason: str
    should_refuse: bool


class QueryRouteResult(BaseModel):
    route: QuestionRoute
    reason: str
    rewritten_queries: list[str] = Field(default_factory=list)


class AnswerVerificationResult(BaseModel):
    supported: bool
    support_score: float
    matched_citations: int
    reason: str


class AgentAnswer(BaseModel):
    question: str
    answer: str
    citations: list[AnswerCitation]
    self_check: SelfCheckResult
    retrieval: list[RetrievalCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcrPageResult(BaseModel):
    page: int
    lines: list[str] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class IndexArtifact(BaseModel):
    parsed_document: ParsedDocument
    chunks: list[DocumentChunk]
    embeddings_path: str


class IngestSummary(BaseModel):
    pdf_type: PdfType
    total_documents: int
    total_pages: int
    total_elements: int
    total_chunks: int
    embedding_backend: str
    reranker_backend: str
    artifact_dir: str

