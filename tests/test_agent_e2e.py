from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.config import Settings


def build_test_pdf(output_path: Path, paragraphs: list[str], table_rows: list[list[str]] | None = None) -> Path:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="STSong-Light")
    story = []
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 12))
    if table_rows:
        story.append(
            Table(
                table_rows,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ]
                ),
            )
        )
    doc.build(story)
    return output_path


def test_agent_can_answer_and_refuse(tmp_path: Path) -> None:
    pdf_path = build_test_pdf(
        tmp_path / "attachment.pdf",
        paragraphs=[
            "第一条 服务范围 乙方提供 PDF 解析、知识库构建、问答检索和自检服务。",
            "第二条 服务期限 项目实施期为 2026 年 6 月 1 日至 2026 年 12 月 31 日。",
        ],
        table_rows=[["阶段", "金额"], ["预付款", "48000"], ["验收款", "72000"]],
    )
    artifact_dir = tmp_path / "artifacts"
    settings = Settings(
        pdf_path=pdf_path,
        artifact_dir=artifact_dir,
        embedding_model_path=str(tmp_path / "missing-embedding-model"),
        reranker_model_path=str(tmp_path / "missing-reranker-model"),
        generator_mode="extractive",
    )

    agent = DocumentQaAgent(settings)
    summary = agent.build_index_summary()
    payment_answer = agent.ask("预付款金额是多少？")
    unknown_answer = agent.ask("合同中约定的发票税率是多少？")

    assert summary.total_chunks > 0
    assert payment_answer.self_check.should_refuse is False
    assert any("48000" in citation.snippet or "48000" in payment_answer.answer for citation in payment_answer.citations)
    assert payment_answer.metadata["query_route"]["route"] == "table"
    assert payment_answer.metadata["retrieval_retry"]["used"] is True
    assert payment_answer.metadata["answer_verification"]["supported"] is True

    clause_answer = agent.ask("第二条的服务期限到什么时候？")
    assert clause_answer.metadata["query_route"]["route"] == "clause"
    assert clause_answer.metadata["retrieval_retry"]["used"] is True
    assert clause_answer.metadata["answer_verification"]["supported"] is True

    assert unknown_answer.self_check.should_refuse is True
    assert unknown_answer.metadata["query_route"]["route"] == "high_no_answer_risk"
    assert unknown_answer.metadata["retrieval_retry"]["used"] is True
    assert unknown_answer.metadata["answer_verification"]["supported"] is False


def test_agent_can_ingest_multiple_pdfs_from_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    build_test_pdf(
        input_dir / "finance.pdf",
        paragraphs=["第一条 付款条款 预付款为 48000 元，验收款为 72000 元。"],
        table_rows=[["阶段", "金额"], ["预付款", "48000"], ["验收款", "72000"]],
    )
    build_test_pdf(
        input_dir / "timeline.pdf",
        paragraphs=["第二条 实施期限 项目实施期为 2026 年 6 月 1 日至 2026 年 12 月 31 日。"],
    )

    artifact_dir = tmp_path / "artifacts"
    settings = Settings(
        pdf_path=input_dir,
        artifact_dir=artifact_dir,
        embedding_model_path=str(tmp_path / "missing-embedding-model"),
        reranker_model_path=str(tmp_path / "missing-reranker-model"),
        generator_mode="extractive",
    )

    agent = DocumentQaAgent(settings)
    summary = agent.build_index_summary()
    answer = agent.ask("项目实施期到什么时候结束？")

    assert summary.total_documents == 2
    assert summary.total_chunks > 0
    assert answer.self_check.should_refuse is False
    assert "2026 年 12 月 31 日" in answer.answer
    assert any(citation.source_pdf == "timeline.pdf" for citation in answer.citations)
    assert answer.metadata["query_route"]["route"] in {"general", "clause"}
    assert "answer_verification" in answer.metadata
