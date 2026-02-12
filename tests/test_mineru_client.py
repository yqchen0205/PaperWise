from pathlib import Path

from papersummarizer.mineru_client import MinerUClient
from papersummarizer.models import MinerUParseOptions, UploadFileSpec


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.headers = {}

    def request(self, method, url, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def test_create_upload_batch_and_get_results(tmp_path):
    session = FakeSession(
        responses=[
            FakeResponse(
                200,
                {
                    "code": 0,
                    "data": {
                        "batch_id": "b-1",
                        "file_urls": [
                            {
                                "name": "a.pdf",
                                "data_id": "d-1",
                                "upload_url": "https://upload.example.com/a.pdf",
                            }
                        ],
                    },
                },
            ),
            FakeResponse(
                200,
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "a.pdf",
                                "data_id": "d-1",
                                "state": "done",
                                "full_zip_url": "https://download.example.com/a.zip",
                            }
                        ]
                    },
                },
            ),
            FakeResponse(
                200,
                {
                    "code": 0,
                    "data": {
                        "state": "done",
                        "file_name": "a.pdf",
                        "data_id": "d-1",
                        "full_zip_url": "https://download.example.com/a.zip",
                    },
                },
            ),
        ]
    )

    client = MinerUClient(
        base_url="https://mineru.example.com",
        api_token="token",
        session=session,
    )

    batch = client.create_upload_batch(
        files=[UploadFileSpec(path=tmp_path / "a.pdf", data_id="d-1")],
        options=MinerUParseOptions(),
    )
    assert batch.batch_id == "b-1"
    assert batch.targets[0].upload_url == "https://upload.example.com/a.pdf"

    results = client.get_batch_results("b-1")
    assert results[0].state == "done"
    assert results[0].result_url == "https://download.example.com/a.zip"

    task_result = client.wait_for_task(
        task_id="task-1",
        poll_interval_sec=0.01,
        timeout_sec=1,
    )
    assert task_result.state == "done"
    assert task_result.result_url == "https://download.example.com/a.zip"


def test_create_upload_batch_accepts_plain_url_list(tmp_path):
    session = FakeSession(
        responses=[
            FakeResponse(
                200,
                {
                    "code": 0,
                    "data": {
                        "batch_id": "b-2",
                        "file_urls": [
                            "https://upload.example.com/plain-a.pdf",
                        ],
                    },
                },
            ),
        ]
    )
    client = MinerUClient(
        base_url="https://mineru.example.com",
        api_token="token",
        session=session,
    )

    file_path = tmp_path / "a.pdf"
    file_path.write_bytes(b"%PDF")
    batch = client.create_upload_batch(
        files=[UploadFileSpec(path=file_path, data_id="d-plain")],
        options=MinerUParseOptions(),
    )

    assert batch.batch_id == "b-2"
    assert batch.targets[0].name == "a.pdf"
    assert batch.targets[0].data_id == "d-plain"
    assert batch.targets[0].upload_url == "https://upload.example.com/plain-a.pdf"
