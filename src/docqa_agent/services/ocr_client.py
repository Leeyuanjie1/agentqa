from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from docqa_agent.config import Settings
from docqa_agent.exceptions import ConfigurationError, ParsingError
from docqa_agent.logging_utils import get_logger
from docqa_agent.schemas import OcrPageResult, TableData


logger = get_logger(__name__)


class OcrClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def recognize_page(self, image_bytes: bytes, page: int) -> OcrPageResult:
        if not self.settings.ocr_api_url:
            raise ConfigurationError("OCR API URL is not configured. Set DOCQA_OCR_API_URL before parsing scanned PDFs.")

        if self.settings.ocr_api_url.rstrip("/").endswith("/jobs"):
            return self._recognize_page_via_job(image_bytes=image_bytes, page=page)

        return self._recognize_page_sync(image_bytes=image_bytes, page=page)

    def _recognize_page_sync(self, image_bytes: bytes, page: int) -> OcrPageResult:
        auth_headers = {}
        if self.settings.ocr_api_key:
            auth_headers["Authorization"] = f"Bearer {self.settings.ocr_api_key}"

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        attempt_errors: list[str] = []

        try:
            for attempt in self._request_attempts(image_bytes=image_bytes, image_base64=image_base64, page=page):
                response = requests.post(
                    self.settings.ocr_api_url,
                    headers=attempt["headers"],
                    json=attempt.get("json"),
                    files=attempt.get("files"),
                    data=attempt.get("data"),
                    timeout=self.settings.ocr_timeout,
                )
                if response.ok:
                    return self._parse_response(page=page, data=response.json())

                body_preview = response.text[:300].replace("\n", " ").strip()
                attempt_errors.append(f"{attempt['name']} => {response.status_code}: {body_preview}")
                if response.status_code not in {400, 404, 415, 422}:
                    response.raise_for_status()

            raise ParsingError(
                f"OCR request failed on page {page}. Tried payload variants: {' | '.join(attempt_errors)}"
            )
        except requests.RequestException as exc:
            logger.exception("OCR request failed on page %s", page)
            raise ParsingError(f"OCR request failed on page {page}: {exc}") from exc

    def _recognize_page_via_job(self, image_bytes: bytes, page: int) -> OcrPageResult:
        headers = self._auth_headers()
        data = {
            "model": self.settings.ocr_model,
            "optionalPayload": json.dumps(self._optional_payload()),
        }
        files = {"file": (f"page_{page}.png", image_bytes, "image/png")}

        try:
            job_response = requests.post(
                self.settings.ocr_api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.settings.ocr_timeout,
            )
            if not job_response.ok:
                body_preview = job_response.text[:300].replace("\n", " ").strip()
                raise ParsingError(
                    f"OCR job submission failed on page {page}: {job_response.status_code}: {body_preview}"
                )

            job_payload = job_response.json()
            job_id = self._extract_job_id(job_payload)
            if not job_id:
                raise ParsingError(f"OCR job submission succeeded on page {page} but jobId is missing.")

            deadline = time.monotonic() + self.settings.ocr_timeout
            while time.monotonic() < deadline:
                status_response = requests.get(
                    f"{self.settings.ocr_api_url.rstrip('/')}/{job_id}",
                    headers=headers,
                    timeout=self.settings.ocr_timeout,
                )
                if not status_response.ok:
                    body_preview = status_response.text[:300].replace("\n", " ").strip()
                    raise ParsingError(
                        f"OCR job polling failed on page {page}: {status_response.status_code}: {body_preview}"
                    )

                status_payload = status_response.json()
                status_data = self._extract_data(status_payload)
                state = str(status_data.get("state", "")).lower()
                if state == "done":
                    result_url = self._extract_result_url(status_data)
                    if not result_url:
                        raise ParsingError(
                            f"OCR job completed on page {page} but resultUrl.jsonUrl is missing."
                        )
                    result_response = requests.get(result_url, timeout=self.settings.ocr_timeout)
                    result_response.raise_for_status()
                    return self._parse_job_result(page=page, jsonl_text=result_response.text, raw=status_payload)

                if state == "failed":
                    error_message = status_data.get("errorMsg") or status_data.get("message") or "unknown error"
                    raise ParsingError(f"OCR job failed on page {page}: {error_message}")

                time.sleep(max(self.settings.ocr_poll_interval, 1))

            raise ParsingError(f"OCR job timed out on page {page} after {self.settings.ocr_timeout} seconds.")
        except requests.RequestException as exc:
            logger.exception("OCR job request failed on page %s", page)
            raise ParsingError(f"OCR job request failed on page {page}: {exc}") from exc

    def _request_attempts(self, image_bytes: bytes, image_base64: str, page: int) -> list[dict[str, Any]]:
        auth_headers = self._auth_headers(bearer_case="Bearer")

        return [
            {
                "name": "json:image_base64+page",
                "headers": {"Content-Type": "application/json", **auth_headers},
                "json": {"page": page, "image_base64": image_base64},
            },
            {
                "name": "json:image+page",
                "headers": {"Content-Type": "application/json", **auth_headers},
                "json": {"page": page, "image": image_base64},
            },
            {
                "name": "json:image_base64",
                "headers": {"Content-Type": "application/json", **auth_headers},
                "json": {"image_base64": image_base64},
            },
            {
                "name": "json:image",
                "headers": {"Content-Type": "application/json", **auth_headers},
                "json": {"image": image_base64},
            },
            {
                "name": "multipart:file",
                "headers": auth_headers,
                "data": {"page": str(page)},
                "files": {"file": (f"page_{page}.png", image_bytes, "image/png")},
            },
            {
                "name": "multipart:image",
                "headers": auth_headers,
                "data": {"page": str(page)},
                "files": {"image": (f"page_{page}.png", image_bytes, "image/png")},
            },
        ]

    def _auth_headers(self, bearer_case: str = "bearer") -> dict[str, str]:
        if not self.settings.ocr_api_key:
            return {}
        return {"Authorization": f"{bearer_case} {self.settings.ocr_api_key}"}

    def _optional_payload(self) -> dict[str, bool]:
        return {
            "useDocOrientationClassify": self.settings.ocr_use_doc_orientation_classify,
            "useDocUnwarping": self.settings.ocr_use_doc_unwarping,
            "useChartRecognition": self.settings.ocr_use_chart_recognition,
        }

    @staticmethod
    def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def _extract_job_id(self, payload: dict[str, Any]) -> str:
        data = self._extract_data(payload)
        job_id = data.get("jobId") or data.get("id")
        return str(job_id) if job_id else ""

    @staticmethod
    def _extract_result_url(payload: dict[str, Any]) -> str:
        result_url = payload.get("resultUrl")
        if isinstance(result_url, dict):
            json_url = result_url.get("jsonUrl") or result_url.get("url")
            return str(json_url) if json_url else ""
        if isinstance(result_url, str):
            return result_url
        return ""

    def _parse_job_result(self, page: int, jsonl_text: str, raw: dict[str, Any]) -> OcrPageResult:
        lines: list[str] = []
        tables: list[TableData] = []
        raw_lines: list[dict[str, Any]] = []

        for raw_line in jsonl_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            raw_lines.append(payload)
            result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            layout_results = result.get("layoutParsingResults") or []
            for layout_result in layout_results:
                markdown_block = layout_result.get("markdown") if isinstance(layout_result, dict) else None
                markdown_text = ""
                if isinstance(markdown_block, dict):
                    markdown_text = str(markdown_block.get("text") or "").strip()
                elif isinstance(markdown_block, str):
                    markdown_text = markdown_block.strip()
                if not markdown_text:
                    continue
                lines.extend(self._extract_markdown_lines(markdown_text))
                tables.extend(self._extract_markdown_tables(markdown_text))

        deduped_lines = list(dict.fromkeys(line for line in lines if line))
        deduped_tables = self._dedupe_tables(tables)
        return OcrPageResult(page=page, lines=deduped_lines, tables=deduped_tables, raw={"job": raw, "jsonl": raw_lines})

    def _parse_response(self, page: int, data: dict[str, Any]) -> OcrPageResult:
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        if isinstance(body.get("result"), dict):
            body = body["result"]
        if isinstance(body.get("output"), dict):
            body = body["output"]
        raw_lines = body.get("lines") or body.get("texts") or body.get("rec_texts") or []
        if not raw_lines and isinstance(body.get("result"), list):
            raw_lines = body["result"]
        lines = [self._coerce_text(item) for item in raw_lines if self._coerce_text(item)]

        raw_tables = body.get("tables") or []
        tables = [self._coerce_table(item) for item in raw_tables]

        return OcrPageResult(page=page, lines=lines, tables=[table for table in tables if table], raw=data)

    @staticmethod
    def _coerce_text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("text", "value", "content", "rec_text", "label"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @staticmethod
    def _coerce_table(item: Any) -> TableData | None:
        if isinstance(item, dict):
            if isinstance(item.get("markdown"), str) and item["markdown"].strip():
                return TableData(
                    headers=item.get("headers") or [],
                    rows=item.get("rows") or [],
                    markdown=item["markdown"].strip(),
                )
            rows = item.get("rows") or item.get("cells")
            if isinstance(rows, list) and rows:
                normalized_rows = [
                    [str(cell).strip() for cell in row]
                    for row in rows
                    if isinstance(row, list) and any(str(cell).strip() for cell in row)
                ]
                if normalized_rows:
                    headers = normalized_rows[0]
                    body_rows = normalized_rows[1:]
                    markdown_rows = [headers, ["---"] * len(headers), *body_rows]
                    markdown = "\n".join(
                        "| " + " | ".join(cell for cell in row) + " |" for row in markdown_rows
                    )
                    return TableData(headers=headers, rows=body_rows, markdown=markdown)
        return None

    @staticmethod
    def _extract_markdown_lines(markdown_text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if OcrClient._is_markdown_table_line(line):
                continue
            if line.startswith("!["):
                continue
            line = re.sub(r"^#+\s*", "", line)
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _extract_markdown_tables(markdown_text: str) -> list[TableData]:
        table_blocks: list[list[str]] = []
        current_block: list[str] = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if OcrClient._is_markdown_table_line(line):
                current_block.append(line)
                continue
            if current_block:
                table_blocks.append(current_block)
                current_block = []
        if current_block:
            table_blocks.append(current_block)

        tables: list[TableData] = []
        for block in table_blocks:
            table = OcrClient._parse_markdown_table_block(block)
            if table:
                tables.append(table)
        return tables

    @staticmethod
    def _parse_markdown_table_block(block: list[str]) -> TableData | None:
        rows = [OcrClient._split_markdown_row(line) for line in block if OcrClient._split_markdown_row(line)]
        if len(rows) < 2:
            return None

        headers = rows[0]
        body_rows = rows[1:]
        if body_rows and OcrClient._is_markdown_divider(body_rows[0]):
            body_rows = body_rows[1:]
        if not headers or not body_rows:
            return None

        markdown = "\n".join(block)
        return TableData(headers=headers, rows=body_rows, markdown=markdown)

    @staticmethod
    def _split_markdown_row(line: str) -> list[str]:
        trimmed = line.strip().strip("|")
        if not trimmed:
            return []
        return [cell.strip() for cell in trimmed.split("|")]

    @staticmethod
    def _is_markdown_divider(row: list[str]) -> bool:
        return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in row if cell)

    @staticmethod
    def _is_markdown_table_line(line: str) -> bool:
        return line.startswith("|") and line.count("|") >= 2

    @staticmethod
    def _dedupe_tables(tables: list[TableData]) -> list[TableData]:
        deduped: list[TableData] = []
        seen: set[str] = set()
        for table in tables:
            key = table.markdown.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(table)
        return deduped


def load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()
