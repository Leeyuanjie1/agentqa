from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docqa_agent.exceptions import ConfigurationError


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_pdf_inputs(candidate: Path) -> list[Path]:
    if candidate.exists() and candidate.is_file():
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {candidate}")
        return [candidate]

    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"PDF path must be a file or directory: {candidate}")

    if not candidate.exists() and candidate.suffix.lower() == ".pdf":
        raise FileNotFoundError(f"PDF file not found: {candidate}")

    search_dir = candidate
    if not search_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {search_dir}")

    pdf_files = sorted(path for path in search_dir.rglob("*.pdf") if path.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under directory: {search_dir}")
    return pdf_files


def resolve_pdf_path(candidate: Path) -> Path:
    pdf_files = resolve_pdf_inputs(candidate)
    if len(pdf_files) != 1:
        candidates = ", ".join(str(path) for path in pdf_files[:10])
        raise ConfigurationError(
            f"Expected a single PDF path, but found {len(pdf_files)} PDFs. Candidates: {candidates}"
        )
    return pdf_files[0]
