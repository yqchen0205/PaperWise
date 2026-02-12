"""MinerU API client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from .exceptions import MinerUApiError
from .models import ExtractResultItem, MinerUParseOptions, UploadBatch, UploadFileSpec, UploadTarget


_TERMINAL_STATES = {"done", "failed", "error", "timeout", "expired"}


@dataclass
class _RateLimiter:
    min_interval_sec: float
    _next_ts: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_ts:
            time.sleep(self._next_ts - now)
        self._next_ts = time.monotonic() + self.min_interval_sec


class MinerUClient:
    """Thin API wrapper for MinerU extraction endpoints."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        timeout_sec: int = 60,
        post_qps_limit: int = 5,
        query_qps_limit: int = 20,
        trust_env: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = session or requests.Session()
        self.session.trust_env = trust_env
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }
        )
        self._post_limiter = _RateLimiter(min_interval_sec=1.0 / max(post_qps_limit, 1))
        self._query_limiter = _RateLimiter(min_interval_sec=1.0 / max(query_qps_limit, 1))

    def _build_url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        limiter = self._query_limiter if method.upper() == "GET" else self._post_limiter
        url = self._build_url(path)

        for attempt in range(retries + 1):
            limiter.wait()
            response = self.session.request(
                method=method.upper(),
                url=url,
                json=json_payload,
                timeout=self.timeout_sec,
            )

            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                backoff = 2**attempt
                time.sleep(backoff)
                continue

            if not response.ok:
                raise MinerUApiError(
                    f"MinerU HTTP error {response.status_code} on {path}: {response.text[:500]}"
                )

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise MinerUApiError(f"Non-JSON response from MinerU on {path}") from exc

            code = payload.get("code")
            if code not in (None, 0, 200):
                raise MinerUApiError(
                    f"MinerU API error code={code} on {path}: {payload.get('msg') or payload.get('message')}"
                )

            return payload

        raise MinerUApiError(f"MinerU request failed after retries: {path}")

    def create_extract_task(self, file_url: str, data_id: str, options: MinerUParseOptions) -> str:
        """Create one URL-based extraction task."""

        payload = {
            "url": file_url,
            "data_id": data_id,
            "is_ocr": options.is_ocr,
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language,
            "model_version": options.model_version,
        }
        body = self._request_json("POST", "/api/v4/extract/task", json_payload=payload)
        data = body.get("data") or {}
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise MinerUApiError("MinerU did not return task_id")
        return str(task_id)

    def get_extract_task(self, task_id: str) -> dict[str, Any]:
        """Fetch one URL-based extraction task status."""

        body = self._request_json("GET", f"/api/v4/extract/task/{task_id}")
        return body.get("data") or {}

    def wait_for_task(
        self,
        task_id: str,
        poll_interval_sec: float,
        timeout_sec: int,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> ExtractResultItem:
        """Poll one URL-based extraction task until terminal state."""

        start = time.monotonic()
        attempt = 0

        while True:
            attempt += 1
            data = self.get_extract_task(task_id)
            state = str(data.get("state") or "unknown").lower()
            result_url = (
                data.get("full_zip_url")
                or data.get("zip_url")
                or data.get("result_url")
                or data.get("url")
            )
            item = ExtractResultItem(
                data_id=str(data.get("data_id") or ""),
                file_name=str(data.get("file_name") or data.get("name") or ""),
                state=state,
                result_url=str(result_url) if result_url else None,
                message=str(data.get("msg") or data.get("message") or "") or None,
            )

            # Report progress via callback
            if progress_callback:
                elapsed = int(time.monotonic() - start)
                progress_pct = 100 if item.state in _TERMINAL_STATES else min(90, attempt * 5)
                progress_callback(
                    progress_pct,
                    f"Attempt {attempt} | State: {state} | {elapsed}s"
                )

            if item.state in _TERMINAL_STATES:
                return item

            if time.monotonic() - start > timeout_sec:
                raise MinerUApiError(f"Timed out waiting for MinerU task: {task_id}")

            time.sleep(poll_interval_sec)

    def create_upload_batch(
        self,
        files: list[UploadFileSpec],
        options: MinerUParseOptions,
    ) -> UploadBatch:
        """Request upload URLs for local PDF files."""

        payload = {
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language,
            "model_version": options.model_version,
            "files": [
                {
                    "name": file.path.name,
                    "data_id": file.data_id,
                    "is_ocr": file.is_ocr,
                }
                for file in files
            ],
        }

        body = self._request_json("POST", "/api/v4/file-urls/batch", json_payload=payload)
        data = body.get("data") or {}
        batch_id = data.get("batch_id")
        if not batch_id:
            raise MinerUApiError("MinerU did not return batch_id")

        file_urls = data.get("file_urls") or data.get("upload_urls") or data.get("files") or []
        if not isinstance(file_urls, list) or not file_urls:
            raise MinerUApiError("MinerU did not return upload URLs")

        by_name: dict[str, UploadFileSpec] = {f.path.name: f for f in files}
        targets: list[UploadTarget] = []
        for index, item in enumerate(file_urls):
            if isinstance(item, str):
                matched_file = files[index] if index < len(files) else None
                if not matched_file:
                    continue
                targets.append(
                    UploadTarget(
                        name=matched_file.path.name,
                        data_id=matched_file.data_id,
                        upload_url=item,
                    )
                )
                continue

            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or "")
            upload_url = item.get("upload_url") or item.get("url") or item.get("presigned_url")
            if not name or not upload_url:
                continue
            declared_data_id = str(item.get("data_id") or "")
            matched = by_name.get(name)
            data_id = declared_data_id or (matched.data_id if matched else "")
            targets.append(UploadTarget(name=name, data_id=data_id, upload_url=str(upload_url)))

        if not targets:
            raise MinerUApiError("No valid upload targets from MinerU response")

        return UploadBatch(batch_id=str(batch_id), targets=targets)

    def upload_to_presigned_url(self, upload_url: str, file_path: Path) -> None:
        """Upload a local file to the presigned storage URL."""

        with requests.Session() as upload_session:
            upload_session.trust_env = self.session.trust_env
            with file_path.open("rb") as fp:
                response = upload_session.put(
                    upload_url,
                    data=fp,
                    timeout=self.timeout_sec,
                )

        if response.status_code >= 400:
            raise MinerUApiError(
                f"Upload failed for {file_path.name}: HTTP {response.status_code} {response.text[:200]}"
            )

    def get_batch_results(self, batch_id: str) -> list[ExtractResultItem]:
        """Query extraction results for a batch."""

        body = self._request_json("GET", f"/api/v4/extract-results/batch/{batch_id}")
        data = body.get("data") or {}
        raw_items = data.get("extract_result") or data.get("results") or data.get("files") or []
        if isinstance(raw_items, dict):
            raw_items = [raw_items]

        results: list[ExtractResultItem] = []
        for item in raw_items:
            result_url = (
                item.get("full_zip_url")
                or item.get("zip_url")
                or item.get("result_url")
                or item.get("url")
            )
            results.append(
                ExtractResultItem(
                    data_id=str(item.get("data_id") or ""),
                    file_name=str(item.get("file_name") or item.get("name") or ""),
                    state=str(item.get("state") or "unknown").lower(),
                    result_url=str(result_url) if result_url else None,
                    message=str(item.get("msg") or item.get("message") or "") or None,
                )
            )
        return results

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval_sec: float,
        timeout_sec: int,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> list[ExtractResultItem]:
        """Poll batch endpoint until all files reach terminal states."""

        start = time.monotonic()
        attempt = 0

        while True:
            attempt += 1
            results = self.get_batch_results(batch_id)

            # Report progress via callback
            if progress_callback and results:
                total = len(results)
                done = sum(1 for r in results if r.state in _TERMINAL_STATES)
                elapsed = int(time.monotonic() - start)
                progress_callback(
                    int(done / total * 100) if total > 0 else 0,
                    f"Attempt {attempt} | {done}/{total} done | {elapsed}s"
                )

            if results and all(result.state in _TERMINAL_STATES for result in results):
                return results

            if time.monotonic() - start > timeout_sec:
                raise MinerUApiError(f"Timed out waiting for MinerU batch: {batch_id}")

            time.sleep(poll_interval_sec)

    def download_result_archive(self, result_url: str, target_zip_path: Path) -> Path:
        """Download extraction archive from MinerU."""

        target_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.Session() as download_session:
            download_session.trust_env = self.session.trust_env
            response = download_session.get(result_url, timeout=self.timeout_sec)
        if response.status_code >= 400:
            raise MinerUApiError(
                f"Failed to download result archive: HTTP {response.status_code}"
            )
        target_zip_path.write_bytes(response.content)
        return target_zip_path
