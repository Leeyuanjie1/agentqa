from __future__ import annotations

from pathlib import Path

from docqa_agent.chunking import build_chunks
from docqa_agent.config import Settings
from docqa_agent.exceptions import ParsingError
from docqa_agent.parsers.classifier import classify_pdf
from docqa_agent.parsers.scanned_pdf import parse_scanned_pdf
from docqa_agent.parsers.text_pdf import parse_text_pdf
from docqa_agent.schemas import DocumentChunk, ParsedDocument
from docqa_agent.services.ocr_client import OcrClient


def parse_document(pdf_path: Path, settings: Settings) -> tuple[ParsedDocument, list[DocumentChunk]]:
    classification = classify_pdf(pdf_path=pdf_path, settings=settings)
    low_text_pages = {item.page for item in classification.page_stats if item.is_low_text}
    high_text_pages = {item.page for item in classification.page_stats if not item.is_low_text}

    if classification.pdf_type == "text":
        parsed = parse_text_pdf(pdf_path=pdf_path, pdf_type="text")
    elif classification.pdf_type == "scanned":
        parsed = parse_scanned_pdf(
            pdf_path=pdf_path,
            pdf_type="scanned",
            settings=settings,
            ocr_client=OcrClient(settings),
        )
    else:
        text_part = parse_text_pdf(pdf_path=pdf_path, pdf_type="mixed", pages=high_text_pages)
        scan_part = parse_scanned_pdf(
            pdf_path=pdf_path,
            pdf_type="mixed",
            settings=settings,
            ocr_client=OcrClient(settings),
            pages=low_text_pages,
        )
        merged_elements = sorted(text_part.elements + scan_part.elements, key=lambda item: (item.page, item.element_type))
        parsed = ParsedDocument(
            pdf_path=str(pdf_path),
            pdf_type="mixed",
            total_pages=classification.total_pages,
            elements=merged_elements,
        )

    source_pdf = pdf_path.name
    for element in parsed.elements:
        element.metadata.setdefault("source_pdf", source_pdf)
        element.metadata.setdefault("source_path", str(pdf_path))

    chunks = build_chunks(parsed.elements, settings=settings, source_pdf=source_pdf)
    if not chunks:
        raise ParsingError(f"No extractable content was produced for {pdf_path}. Check OCR configuration or input quality.")
    return parsed, chunks


def parse_documents(pdf_paths: list[Path], settings: Settings, root_path: Path) -> tuple[ParsedDocument, list[DocumentChunk]]:
    parsed_documents: list[ParsedDocument] = []
    all_chunks: list[DocumentChunk] = []

    for doc_index, pdf_path in enumerate(pdf_paths, start=1):
        parsed_document, chunks = parse_document(pdf_path=pdf_path, settings=settings)
        chunk_prefix = f"doc{doc_index}"
        source_pdf = pdf_path.name

        for chunk in chunks:
            chunk.chunk_id = f"{chunk_prefix}-{chunk.chunk_id}"
            chunk.source_pdf = source_pdf
            chunk.metadata.setdefault("source_path", str(pdf_path))
            chunk.metadata.setdefault("doc_index", doc_index)

        parsed_document.source_documents = [source_pdf]
        parsed_document.metadata = {
            "source_path": str(pdf_path),
            "doc_index": doc_index,
        }

        parsed_documents.append(parsed_document)
        all_chunks.extend(chunks)

    pdf_types = {document.pdf_type for document in parsed_documents}
    aggregated_pdf_type = pdf_types.pop() if len(pdf_types) == 1 else "mixed"
    combined_elements = []
    for document in parsed_documents:
        combined_elements.extend(document.elements)

    combined_parsed_document = ParsedDocument(
        pdf_path=str(root_path),
        pdf_type=aggregated_pdf_type,
        total_pages=sum(document.total_pages for document in parsed_documents),
        elements=combined_elements,
        source_documents=[document.source_documents[0] for document in parsed_documents if document.source_documents],
        metadata={
            "document_count": len(parsed_documents),
            "source_paths": [document.metadata.get("source_path") for document in parsed_documents],
        },
    )
    return combined_parsed_document, all_chunks
