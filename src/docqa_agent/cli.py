from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich import print

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.config import get_settings
from docqa_agent.logging_utils import configure_logging


app = typer.Typer(help="Minimal document QA agent", no_args_is_help=True)


@app.command()
def ingest(
    pdf_path: Optional[str] = typer.Option(default=None, help="Path to the PDF file."),
    artifact_dir: Optional[str] = typer.Option(default=None, help="Directory to store parsed artifacts and index."),
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    agent = DocumentQaAgent(settings)
    summary = agent.build_index_summary(
        pdf_path=Path(pdf_path) if pdf_path else None,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
    )
    print(summary.model_dump(mode="json"))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask."),
    artifact_dir: Optional[str] = typer.Option(default=None, help="Index artifact directory."),
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    agent = DocumentQaAgent(settings)
    answer = agent.ask(question=question, artifact_dir=Path(artifact_dir) if artifact_dir else None)
    print(answer.model_dump(mode="json"))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 9060) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run("docqa_agent.api:app", host=host, port=port, reload=False)
