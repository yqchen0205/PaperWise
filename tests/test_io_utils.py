from pathlib import Path

from papersummarizer.io_utils import build_output_path


def test_build_output_path_preserves_multi_dot_filename_for_file_input(tmp_path):
    pdf_path = Path("papers/2404.13501.pdf")
    input_root = pdf_path
    output_path = build_output_path(
        pdf_path=pdf_path,
        input_root=input_root,
        output_root=tmp_path,
        category="metadata",
        new_suffix=".json",
    )

    assert output_path.name == "2404.13501.json"


def test_build_output_path_preserves_multi_dot_filename_for_dir_input(tmp_path):
    input_root = Path("papers")
    pdf_path = Path("papers/sub/2404.13501.pdf")
    output_path = build_output_path(
        pdf_path=pdf_path,
        input_root=input_root,
        output_root=tmp_path,
        category="summaries",
        new_suffix=".md",
    )

    assert output_path == tmp_path / "summaries" / "sub" / "2404.13501.md"
