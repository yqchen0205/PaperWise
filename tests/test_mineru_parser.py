import zipfile
from pathlib import Path

from papersummarizer.mineru_parser import MinerUPdfParser
from papersummarizer.models import ExtractResultItem, MinerUParseOptions, UploadBatch, UploadTarget


class FakeMinerUClient:
    def __init__(self):
        self.uploaded_paths = []

    def create_upload_batch(self, files, options):
        file = files[0]
        return UploadBatch(
            batch_id="batch-1",
            targets=[
                UploadTarget(
                    name=file.path.name,
                    data_id=file.data_id,
                    upload_url="https://upload.example.com/file",
                )
            ],
        )

    def upload_to_presigned_url(self, upload_url: str, file_path: Path):
        self.uploaded_paths.append(file_path)

    def wait_for_batch(self, batch_id: str, poll_interval_sec: float, timeout_sec: int):
        return [
            ExtractResultItem(
                data_id="",
                file_name="test.pdf",
                state="done",
                result_url="https://download.example.com/result.zip",
                message=None,
            )
        ]

    def create_extract_task(self, file_url: str, data_id: str, options: MinerUParseOptions):
        return "task-1"

    def wait_for_task(self, task_id: str, poll_interval_sec: float, timeout_sec: int):
        return ExtractResultItem(
            data_id="",
            file_name="remote.pdf",
            state="done",
            result_url="https://download.example.com/result.zip",
            message=None,
        )

    def download_result_archive(self, result_url: str, target_zip_path: Path):
        target_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_zip_path, "w") as zf:
            zf.writestr("full.md", "# Parsed Content\n\nhello")
        return target_zip_path


def test_parse_pdf_success(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parser = MinerUPdfParser(
        client=FakeMinerUClient(),
        parse_options=MinerUParseOptions(),
        poll_interval_sec=0.1,
        poll_timeout_sec=5,
    )

    parsed = parser.parse_pdf(pdf_path=pdf_path, artifact_dir=tmp_path / "artifacts")

    assert parsed.pdf_path == pdf_path
    assert "Parsed Content" in parsed.markdown_text
    assert (tmp_path / "artifacts" / "mineru_result.zip").exists()


def test_parse_pdf_url_success(tmp_path):
    parser = MinerUPdfParser(
        client=FakeMinerUClient(),
        parse_options=MinerUParseOptions(),
        poll_interval_sec=0.1,
        poll_timeout_sec=5,
    )

    parsed = parser.parse_pdf_url(
        pdf_url="https://example.com/remote.pdf",
        artifact_dir=tmp_path / "artifacts_url",
        file_name="remote.pdf",
    )

    assert parsed.pdf_path == Path("remote.pdf")
    assert "Parsed Content" in parsed.markdown_text
