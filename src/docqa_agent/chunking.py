from __future__ import annotations

import re
from typing import Iterable

from docqa_agent.config import Settings
from docqa_agent.schemas import DocumentChunk, DocumentElement


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= chunk_size:
        yield normalized
        return

    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        yield normalized[start:end]
        if end == len(normalized):
            break
        start = max(end - overlap, 0)


def build_chunks(
    elements: list[DocumentElement],
    settings: Settings,
    source_pdf: str,
    chunk_id_prefix: str = "",
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    prefix = f"{chunk_id_prefix}-" if chunk_id_prefix else ""
    for element_index, element in enumerate(elements, start=1):
        base_metadata = dict(element.metadata)
        if element.element_type == "table" and element.table:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{prefix}p{element.page}-t{element_index}",
                    source_pdf=source_pdf,
                    page=element.page,
                    text=element.table.markdown,
                    source_type="table",
                    clause_id=element.clause_id,
                    table_markdown=element.table.markdown,
                    metadata=base_metadata,
                )
            )
            continue

        for segment_index, segment in enumerate(
            _split_text(element.text, settings.chunk_size, settings.chunk_overlap), start=1
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{prefix}p{element.page}-e{element_index}-c{segment_index}",
                    source_pdf=source_pdf,
                    page=element.page,
                    text=segment,
                    source_type=element.element_type,
                    clause_id=element.clause_id,
                    metadata=base_metadata,
                )
            )
    return chunks
