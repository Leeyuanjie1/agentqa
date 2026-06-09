from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.config import get_settings
from docqa_agent.exceptions import ConfigurationError, IndexNotFoundError, ParsingError
from docqa_agent.logging_utils import configure_logging


settings = get_settings()
configure_logging(settings.log_level)
app = FastAPI(title="Document QA Agent", version="0.1.0", default_response_class=ORJSONResponse)
agent = DocumentQaAgent(settings)


class AskRequest(BaseModel):
    question: str
    artifact_dir: str | None = None


class IngestRequest(BaseModel):
    pdf_path: str | None = None
    artifact_dir: str | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "artifact_dir": str(settings.artifact_dir),
    }


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, object]:
    try:
        summary = agent.build_index_summary(
            pdf_path=Path(payload.pdf_path) if payload.pdf_path else None,
            artifact_dir=Path(payload.artifact_dir) if payload.artifact_dir else None,
        )
    except (ConfigurationError, ParsingError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summary.model_dump(mode="json")


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    try:
        answer = agent.ask(
            question=payload.question,
            artifact_dir=Path(payload.artifact_dir) if payload.artifact_dir else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IndexNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ConfigurationError, ParsingError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return answer.model_dump(mode="json")
