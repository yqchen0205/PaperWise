import json

from papersummarizer.summary_evidence_enricher import SummaryEvidenceEnricher


def test_enricher_attaches_figure_and_table_assets(tmp_path):
    artifact_dir = tmp_path / "outputs" / "mineru_artifacts" / "paper"
    extract_dir = artifact_dir / "mineru_extract"
    images_dir = extract_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for image_name in ["fig2a.jpg", "fig2b.jpg", "fig2_caption.jpg", "table1.jpg"]:
        (images_dir / image_name).write_bytes(b"jpeg")

    content_list = [
        {
            "type": "image",
            "img_path": "images/fig2a.jpg",
            "image_caption": ["(a) BudgetThinker on MATH-500"],
            "image_footnote": [],
            "page_idx": 6,
        },
        {
            "type": "image",
            "img_path": "images/fig2b.jpg",
            "image_caption": ["(b) ThinkPrune on MATH-500"],
            "image_footnote": [],
            "page_idx": 6,
        },
        {
            "type": "image",
            "img_path": "images/fig2_caption.jpg",
            "image_caption": ["Figure 2 | Pass@1 accuracy under different budgets."],
            "image_footnote": [],
            "page_idx": 6,
        },
        {
            "type": "table",
            "img_path": "images/table1.jpg",
            "table_caption": ["Table 1 | AIME 2024 results with different budgets."],
            "table_footnote": [],
            "table_body": "<table><tr><td>B=2000</td><td>16.25</td></tr></table>",
            "page_idx": 7,
        },
    ]
    (extract_dir / "sample_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "summaries" / "paper.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_text = "\n".join(
        [
            "[SECTION-E] 模块三：效果验证 (The Evidence)",
            "- MATH-500: gains are visible (Figure 2a/b).",
            "- AIME: improvement is modest (Table 1).",
        ]
    )

    enricher = SummaryEvidenceEnricher()
    enriched_text, coverage = enricher.enrich_summary(
        summary_text=summary_text,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )

    assert "Evidence attachment:" not in enriched_text
    assert "![Figure 2a](paper/images/fig2a.jpg)" in enriched_text
    assert "![Figure 2b](paper/images/fig2b.jpg)" in enriched_text
    assert "![Table 1](paper/images/table1.jpg)" in enriched_text
    assert "Raw table data (MinerU):" not in enriched_text
    assert "<table><tr><td>B=2000</td><td>16.25</td></tr></table>" not in enriched_text

    assert (summary_path.parent / "paper" / "images" / "fig2a.jpg").exists()
    assert (summary_path.parent / "paper" / "images" / "fig2b.jpg").exists()
    assert (summary_path.parent / "paper" / "images" / "table1.jpg").exists()

    assert coverage["detected_count"] == 2
    assert coverage["attached_count"] == 2
    assert coverage["missing_count"] == 0
    assert coverage["is_complete"] is True


def test_enricher_reports_missing_refs_when_no_assets_available(tmp_path):
    artifact_dir = tmp_path / "outputs" / "mineru_artifacts" / "paper"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tmp_path / "outputs" / "summaries" / "paper.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_text = "Budget gains are shown in Figure 2 and Table 1."
    enricher = SummaryEvidenceEnricher()
    enriched_text, coverage = enricher.enrich_summary(
        summary_text=summary_text,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )

    assert enriched_text.strip() == summary_text
    assert coverage["detected_count"] == 2
    assert coverage["attached_count"] == 0
    assert coverage["missing_count"] == 2
    assert coverage["is_complete"] is False


def test_enricher_is_idempotent_for_existing_attachments(tmp_path):
    artifact_dir = tmp_path / "outputs" / "mineru_artifacts" / "paper"
    extract_dir = artifact_dir / "mineru_extract"
    images_dir = extract_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / "fig2a.jpg").write_bytes(b"jpeg")
    content_list = [
        {
            "type": "image",
            "img_path": "images/fig2a.jpg",
            "image_caption": ["Figure 2 | Pass@1 accuracy under different budgets."],
            "image_footnote": [],
            "page_idx": 6,
        }
    ]
    (extract_dir / "sample_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "summaries" / "paper.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    base_summary = "Reference line (Figure 2)."

    enricher = SummaryEvidenceEnricher()
    once_text, _ = enricher.enrich_summary(
        summary_text=base_summary,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )
    twice_text, coverage = enricher.enrich_summary(
        summary_text=once_text,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )

    assert once_text == twice_text
    assert "![Figure 2](paper/images/fig2a.jpg)" in once_text
    assert once_text.count("![Figure 2](paper/images/fig2a.jpg)") == 1
    assert coverage["attached_count"] == 1
    assert coverage["missing_count"] == 0


def test_enricher_copies_absolute_local_paths_to_summary_assets(tmp_path):
    artifact_dir = tmp_path / "outputs" / "mineru_artifacts" / "paper"
    extract_dir = artifact_dir / "mineru_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    external_dir = tmp_path / "external_images"
    external_dir.mkdir(parents=True, exist_ok=True)
    external_image = external_dir / "fig9.jpg"
    external_image.write_bytes(b"jpeg")

    content_list = [
        {
            "type": "image",
            "img_path": str(external_image),
            "image_caption": ["Figure 9 | Absolute path image source."],
            "image_footnote": [],
            "page_idx": 1,
        }
    ]
    (extract_dir / "sample_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "summaries" / "paper.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    enricher = SummaryEvidenceEnricher()
    enriched_text, coverage = enricher.enrich_summary(
        summary_text="Result shown in Figure 9.",
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )

    assert "![Figure 9](paper/external/fig9.jpg)" in enriched_text
    assert (summary_path.parent / "paper" / "external" / "fig9.jpg").exists()
    assert coverage["attached_count"] == 1
