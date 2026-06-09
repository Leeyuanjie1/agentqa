from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fitz
import pdfplumber

from docqa_agent.schemas import DocumentElement, ParsedDocument, TableData


CLAUSE_PATTERNS = [
    re.compile(r"^(第[一二三四五六七八九十百千0-9]+条)"),
    re.compile(r"^((?:\d+\.)+\d+|\d+)(?:[\s、.)]|$)"),
]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_clause_id(text: str) -> str | None:
    for pattern in CLAUSE_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1)
    return None


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    headers = rows[0]
    divider = ["---"] * len(headers)
    body_rows = rows[1:] if len(rows) > 1 else []
    markdown_rows = [headers, divider, *body_rows]
    return "\n".join("| " + " | ".join(cell or "" for cell in row) + " |" for row in markdown_rows)


FINANCIAL_HEADER_HINTS = ("被投资单位名称", "账面价值", "本期增加", "本期减少", "减值准备")
NUMBER_TOKEN_PATTERN = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-")


def _extract_financial_table_elements(block_texts: list[str], page_number: int) -> list[DocumentElement]:
    if not block_texts:
        return []

    header_index = next(
        (index for index, text in enumerate(block_texts) if sum(hint in text for hint in FINANCIAL_HEADER_HINTS) >= 2),
        None,
    )
    if header_index is None:
        return []

    merged_lines: list[str] = []
    index = header_index
    while index < len(block_texts):
        current = block_texts[index]
        next_text = block_texts[index + 1] if index + 1 < len(block_texts) else ""

        current_has_number = bool(NUMBER_TOKEN_PATTERN.search(current))
        next_has_number = bool(NUMBER_TOKEN_PATTERN.search(next_text))
        if current and not current_has_number and next_text and next_has_number:
            merged_lines.append(f"{current} {next_text}".strip())
            index += 2
            continue

        merged_lines.append(current)
        index += 1

    headers = ["被投资单位名称", "数值1", "数值2", "数值3", "数值4", "减值准备"]
    elements: list[DocumentElement] = []
    for line in merged_lines:
        if sum(hint in line for hint in FINANCIAL_HEADER_HINTS) >= 2:
            continue

        values = NUMBER_TOKEN_PATTERN.findall(line)
        if len(values) < 3:
            continue

        first_value = values[0]
        split_index = line.find(first_value)
        if split_index <= 0:
            continue

        entity_name = _clean_text(line[:split_index])
        if len(entity_name) < 2:
            continue

        normalized_values = values[:5]
        while len(normalized_values) < 5:
            normalized_values.append("")
        row = [entity_name, *normalized_values]
        markdown = _rows_to_markdown([headers, row])
        elements.append(
            DocumentElement(
                page=page_number,
                element_type="table",
                text=markdown,
                table=TableData(headers=headers, rows=[row], markdown=markdown),
                metadata={"synthetic_table": True, "entity_name": entity_name},
            )
        )

    return elements


def parse_text_pdf(pdf_path: Path, pdf_type: str, pages: Iterable[int] | None = None) -> ParsedDocument:
    selected_pages = set(pages or [])
    use_page_filter = bool(selected_pages)
    elements: list[DocumentElement] = []

    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        for page_index in range(len(fitz_doc)):
            page_number = page_index + 1
            if use_page_filter and page_number not in selected_pages:
                continue

            page = fitz_doc.load_page(page_index)
            blocks = page.get_text("blocks") or []
            page_block_texts: list[str] = []
            for block in blocks:
                text = _clean_text(block[4] if len(block) > 4 else "")
                if not text:
                    continue
                page_block_texts.append(text)
                clause_id = detect_clause_id(text)
                elements.append(
                    DocumentElement(
                        page=page_number,
                        element_type="clause" if clause_id else "paragraph",
                        text=text,
                        clause_id=clause_id,
                    )
                )

            plumber_page = plumber_doc.pages[page_index]
            for raw_table in plumber_page.extract_tables() or []:
                normalized_rows = [
                    [_clean_text(cell or "") for cell in row]
                    for row in raw_table
                    if row and any((cell or "").strip() for cell in row)
                ]
                if not normalized_rows:
                    continue
                table = TableData(
                    headers=normalized_rows[0],
                    rows=normalized_rows[1:],
                    markdown=_rows_to_markdown(normalized_rows),
                )
                elements.append(
                    DocumentElement(
                        page=page_number,
                        element_type="table",
                        text=table.markdown,
                        table=table,
                    )
                )

            existing_table_texts = {element.text for element in elements if element.page == page_number and element.element_type == "table"}
            for synthetic_table in _extract_financial_table_elements(page_block_texts, page_number):
                if synthetic_table.text not in existing_table_texts:
                    elements.append(synthetic_table)

    return ParsedDocument(
        pdf_path=str(pdf_path),
        pdf_type=pdf_type,
        total_pages=len(fitz.open(pdf_path)),
        elements=elements,
    )
