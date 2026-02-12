"""High-level PDF parser built on MinerU APIs."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import MinerUParseError
from .mineru_client import MinerUClient
from .models import ExtractResultItem, MinerUParseOptions, ParsedPaper, UploadFileSpec

if TYPE_CHECKING:
    from collections.abc import Callable


class MinerUPdfParser:
    """Parse local PDFs into markdown text using MinerU."""

    def __init__(
        self,
        client: MinerUClient,
        parse_options: MinerUParseOptions,
        poll_interval_sec: float = 2.0,
        poll_timeout_sec: int = 900,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.client = client
        self.parse_options = parse_options
        self.poll_interval_sec = poll_interval_sec
        self.poll_timeout_sec = poll_timeout_sec
        self.progress_callback = progress_callback

    def parse_pdf(self, pdf_path: Path, artifact_dir: Path) -> ParsedPaper:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise MinerUParseError(f"PDF does not exist: {pdf_path}")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        data_id = uuid.uuid4().hex

        file_spec = UploadFileSpec(path=pdf_path, data_id=data_id, is_ocr=self.parse_options.is_ocr)
        batch = self.client.create_upload_batch(files=[file_spec], options=self.parse_options)

        target = None
        for candidate in batch.targets:
            if candidate.data_id == data_id or candidate.name == pdf_path.name:
                target = candidate
                break

        if target is None:
            raise MinerUParseError(
                f"Cannot find upload target for {pdf_path.name} in MinerU batch response"
            )

        self.client.upload_to_presigned_url(target.upload_url, pdf_path)

        def _progress_wrapper(pct: int, details: str) -> None:
            if self.progress_callback:
                self.progress_callback("parsing", details)

        results = self.client.wait_for_batch(
            batch_id=batch.batch_id,
            poll_interval_sec=self.poll_interval_sec,
            timeout_sec=self.poll_timeout_sec,
            progress_callback=_progress_wrapper,
        )

        result = None
        for item in results:
            if item.data_id == data_id or item.file_name == pdf_path.name:
                result = item
                break

        if result is None:
            raise MinerUParseError(f"MinerU returned no result item for {pdf_path.name}")

        return self._materialize_result(pdf_path=pdf_path, result=result, artifact_dir=artifact_dir)

    def parse_pdf_url(self, pdf_url: str, artifact_dir: Path, file_name: str | None = None) -> ParsedPaper:
        """Parse a remote PDF URL through /api/v4/extract/task endpoints."""

        artifact_dir.mkdir(parents=True, exist_ok=True)
        data_id = uuid.uuid4().hex
        task_id = self.client.create_extract_task(
            file_url=pdf_url,
            data_id=data_id,
            options=self.parse_options,
        )

        def _progress_wrapper(pct: int, details: str) -> None:
            if self.progress_callback:
                self.progress_callback("parsing", details)

        result = self.client.wait_for_task(
            task_id=task_id,
            poll_interval_sec=self.poll_interval_sec,
            timeout_sec=self.poll_timeout_sec,
            progress_callback=_progress_wrapper,
        )

        resolved_name = file_name or result.file_name or f"{data_id}.pdf"
        return self._materialize_result(
            pdf_path=Path(resolved_name),
            result=result,
            artifact_dir=artifact_dir,
        )

    def _materialize_result(
        self,
        pdf_path: Path,
        result: ExtractResultItem,
        artifact_dir: Path,
    ) -> ParsedPaper:
        if result.state != "done":
            raise MinerUParseError(
                f"MinerU parse failed for {pdf_path.name}: state={result.state}, msg={result.message}"
            )

        if not result.result_url:
            raise MinerUParseError(f"MinerU parse finished but has no result URL for {pdf_path.name}")

        archive_path = artifact_dir / "mineru_result.zip"
        extract_dir = artifact_dir / "mineru_extract"
        self.client.download_result_archive(result.result_url, archive_path)

        if extract_dir.exists():
            for path in sorted(extract_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(extract_dir)

        markdown_text = self._read_best_text(extract_dir)
        if not markdown_text.strip():
            raise MinerUParseError(f"No readable text extracted for {pdf_path.name}")

        return ParsedPaper(pdf_path=pdf_path, markdown_text=markdown_text, artifact_dir=artifact_dir)

    def _read_best_text(self, root: Path) -> str:
        markdown_files = sorted(
            root.rglob("*.md"),
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True,
        )
        for markdown_file in markdown_files:
            text = markdown_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text

        json_candidates = [
            p
            for p in root.rglob("*.json")
            if p.name.lower() in {"content_list.json", "content.json", "result.json"}
        ]
        for json_file in json_candidates:
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            text = self._collect_text(payload).strip()
            if text:
                return text

        txt_files = sorted(root.rglob("*.txt"))
        if txt_files:
            chunks = [p.read_text(encoding="utf-8", errors="ignore").strip() for p in txt_files]
            text = "\n\n".join(chunk for chunk in chunks if chunk)
            if text:
                return text

        return ""

    def _collect_text(self, payload: Any) -> str:
        chunks: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_lower = str(key).lower()
                    if key_lower in {"text", "content", "md", "markdown"} and isinstance(value, str):
                        chunks.append(value)
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)
            elif isinstance(node, str):
                chunks.append(node)

        _walk(payload)
        return "\n".join(chunk for chunk in chunks if chunk.strip())
