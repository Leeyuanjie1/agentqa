from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from docqa_agent.config import Settings
from docqa_agent.schemas import PdfType


@dataclass
class PageStats:
    page: int
    char_count: int
    is_low_text: bool


@dataclass
class PdfClassification:
    pdf_type: PdfType
    total_pages: int
    page_stats: list[PageStats]


def classify_pdf(pdf_path: Path, settings: Settings) -> PdfClassification:
    with fitz.open(pdf_path) as doc:
        stats: list[PageStats] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            char_count = len(text.strip())
            stats.append(
                PageStats(
                    page=page_index + 1,
                    char_count=char_count,
                    is_low_text=char_count < settings.min_text_chars_per_page,
                )
            )

    total_pages = len(stats)
    low_text_pages = sum(1 for item in stats if item.is_low_text)
    ratio = low_text_pages / max(total_pages, 1)

    if ratio == 0:
        pdf_type: PdfType = "text"
    elif ratio >= settings.scan_page_ratio_threshold:
        pdf_type = "scanned"
    else:
        pdf_type = "mixed"

    return PdfClassification(pdf_type=pdf_type, total_pages=total_pages, page_stats=stats)
