from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

from docqa_agent.config import Settings
from docqa_agent.exceptions import ParsingError
from docqa_agent.logging_utils import get_logger
from docqa_agent.parsers.text_pdf import detect_clause_id
from docqa_agent.schemas import DocumentElement, ParsedDocument
from docqa_agent.services.ocr_client import OcrClient


logger = get_logger(__name__)


def _render_page_image(page: fitz.Page, dpi: int) -> bytes:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_scanned_pdf(
    pdf_path: Path,
    pdf_type: str,
    settings: Settings,
    ocr_client: OcrClient,
    pages: Iterable[int] | None = None,
) -> ParsedDocument:
    selected_pages = set(pages or [])
    use_page_filter = bool(selected_pages)
    elements: list[DocumentElement] = []
    failed_pages: list[dict[str, str | int]] = []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_index in range(total_pages):
            page_number = page_index + 1
            if use_page_filter and page_number not in selected_pages:
                continue

            page = doc.load_page(page_index)
            image_bytes = _render_page_image(page, settings.render_dpi)
            try:
                ocr_result = ocr_client.recognize_page(image_bytes=image_bytes, page=page_number)
            except ParsingError as exc:
                fallback_text = (page.get_text("text") or "").strip()
                failed_pages.append({"page": page_number, "error": str(exc)})
                logger.warning("OCR failed on page %s of %s: %s", page_number, pdf_path.name, exc)
                if fallback_text:
                    clause_id = detect_clause_id(fallback_text)
                    elements.append(
                        DocumentElement(
                            page=page_number,
                            element_type="clause" if clause_id else "paragraph",
                            text=fallback_text,
                            clause_id=clause_id,
                            metadata={"ocr_failed": True, "ocr_error": str(exc), "fallback_text": True},
                        )
                    )
                    continue
                if settings.ocr_fail_fast:
                    raise
                continue

            page_text = " ".join(line.strip() for line in ocr_result.lines if line.strip())
            if page_text:
                clause_id = detect_clause_id(page_text)
                elements.append(
                    DocumentElement(
                        page=page_number,
                        element_type="clause" if clause_id else "paragraph",
                        text=page_text,
                        clause_id=clause_id,
                        metadata={"ocr": True},
                    )
                )

            for table in ocr_result.tables:
                elements.append(
                    DocumentElement(
                        page=page_number,
                        element_type="table",
                        text=table.markdown,
                        table=table,
                        metadata={"ocr": True},
                    )
                )

    return ParsedDocument(
        pdf_path=str(pdf_path),
        pdf_type=pdf_type,
        total_pages=total_pages,
        elements=elements,
        metadata={"ocr_failed_pages": failed_pages},
    )
