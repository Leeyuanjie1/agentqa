from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pdf_path: Path = Field(default=Path("data/input"))
    artifact_dir: Path = Field(default=Path("data/artifacts/attachment"))
    app_name: str = Field(default="docqa-agent")
    environment: str = Field(default="dev")
    log_level: str = Field(default="INFO")

    embedding_model_path: str = Field(default="models/bge-m3")
    reranker_model_path: str = Field(default="models/bge-reranker-large")

    generator_mode: str = Field(default="extractive")
    llm_base_url: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)

    ocr_api_url: str | None = Field(default=None)
    ocr_api_key: str | None = Field(default=None)
    ocr_model: str = Field(default="PaddleOCR-VL-1.6")
    ocr_timeout: int = Field(default=60)
    ocr_poll_interval: int = Field(default=5)
    ocr_use_doc_orientation_classify: bool = Field(default=False)
    ocr_use_doc_unwarping: bool = Field(default=False)
    ocr_use_chart_recognition: bool = Field(default=False)
    ocr_fail_fast: bool = Field(default=False)

    top_k: int = Field(default=8)
    vector_weight: float = Field(default=0.55)
    bm25_weight: float = Field(default=0.45)
    rerank_threshold: float = Field(default=0.15)
    refuse_threshold: float = Field(default=0.08)

    render_dpi: int = Field(default=180)
    min_text_chars_per_page: int = Field(default=80)
    scan_page_ratio_threshold: float = Field(default=0.6)
    chunk_size: int = Field(default=380)
    chunk_overlap: int = Field(default=60)
    max_citation_count: int = Field(default=3)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
