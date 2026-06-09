from docqa_agent.chunking import build_chunks
from docqa_agent.config import Settings
from docqa_agent.schemas import DocumentElement, TableData


def test_build_chunks_keeps_table_markdown_and_clause_id() -> None:
    settings = Settings(chunk_size=20, chunk_overlap=5)
    elements = [
        DocumentElement(page=1, element_type="clause", text="第一条 服务范围 乙方提供 PDF 解析与问答服务。", clause_id="第一条"),
        DocumentElement(
            page=2,
            element_type="table",
            text="| 阶段 | 金额 |",
            table=TableData(headers=["阶段", "金额"], rows=[["预付款", "48000"]], markdown="| 阶段 | 金额 |\n| --- | --- |\n| 预付款 | 48000 |"),
        ),
    ]

    chunks = build_chunks(elements, settings, source_pdf="sample.pdf")

    assert any(chunk.clause_id == "第一条" for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if chunk.source_type == "table"]
    assert len(table_chunks) == 1
    assert "48000" in table_chunks[0].table_markdown
    assert all(chunk.source_pdf == "sample.pdf" for chunk in chunks)
