"""Filesystem helpers for pipeline orchestration."""

from __future__ import annotations

from pathlib import Path


def _is_file_input(path: Path) -> bool:
    return path.is_file() or path.suffix.lower() == ".pdf"


def discover_pdf_paths(input_path: Path, max_files: int | None = None) -> list[Path]:
    if input_path.is_file():
        paths = [input_path] if input_path.suffix.lower() == ".pdf" else []
    else:
        paths = sorted(input_path.rglob("*.pdf"))

    if max_files is not None:
        return paths[:max_files]
    return paths


def build_output_path(
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    category: str,
    new_suffix: str,
) -> Path:
    relative = (
        Path(pdf_path.name)
        if _is_file_input(input_root)
        else pdf_path.relative_to(input_root)
    )
    output_path = output_root / category / relative
    output_path = output_path.with_suffix(new_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def build_artifact_dir(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    if _is_file_input(input_root):
        relative = Path(pdf_path.stem)
    else:
        relative = pdf_path.relative_to(input_root).with_suffix("")

    artifact_dir = output_root / "mineru_artifacts" / relative
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
