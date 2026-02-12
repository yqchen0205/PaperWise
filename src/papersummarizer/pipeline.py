"""End-to-end pipeline: PDF parsing then LLM summarization."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from tqdm import tqdm

from .io_utils import build_artifact_dir, build_output_path, discover_pdf_paths
from .mineru_parser import MinerUPdfParser
from .models import PipelineResult
from .openai_summarizer import OpenAISummarizer
from .summary_evidence_enricher import SummaryEvidenceEnricher

if TYPE_CHECKING:
    from .progress import RichProgressTracker, StageUpdater

SUMMARY_LAYER_HEADINGS = {
    "hook_tldr": "## 1) TL;DR",
    "motivation_gap": "## 2) Motivation & Gap",
    "method_mechanism": "## 3) Method & Mechanism",
    "proof_results": "## 4) Proof & Results",
    "insights_decision": "## 5) Insights & Decision",
}

LAYER_INDEX_TO_KEY = {
    "1": "hook_tldr",
    "2": "motivation_gap",
    "3": "method_mechanism",
    "4": "proof_results",
    "5": "insights_decision",
}

SEPARATOR_LINE_PATTERN = re.compile(r"^\s*---\s*$", re.MULTILINE)
TITLE_LINE_PATTERN = re.compile(r"^\s*#\s+.+$", re.MULTILINE)
ORIGINAL_TITLE_LINE_PATTERN = re.compile(
    r"^\s*>\s*(?:原论文标题|标题)\s*[:：]\s*\S+", re.MULTILINE
)
FIGURE_TABLE_REF_PATTERN = re.compile(
    r"(Figure|Fig\.?|Table|图|表)\s*\d+", re.IGNORECASE
)
ONE_LINER_PATTERN = re.compile(r"(一句话|One-?Liner|TL;DR|省流)", re.IGNORECASE)
PAIN_POINT_PATTERN = re.compile(r"(痛点|瓶颈|缺陷|问题)", re.IGNORECASE)
GAP_PATTERN = re.compile(r"(研究空白|gap|尚未解决|之前没人解决)", re.IGNORECASE)
INSIGHT_PATTERN = re.compile(r"(核心洞察|key insight|洞察)", re.IGNORECASE)
SOTA_PATTERN = re.compile(
    r"(SOTA|state-of-the-art|相对\s*.+\s*提升|比\s*.+\s*提升)", re.IGNORECASE
)
ABLATION_PATTERN = re.compile(r"(消融|ablation)", re.IGNORECASE)
VISUAL_CASE_PATTERN = re.compile(
    r"(可视化|案例|bad case|good case|错误案例)", re.IGNORECASE
)
LIMITATION_PATTERN = re.compile(r"(局限|限制|风险|失败模式)", re.IGNORECASE)
FUTURE_WORK_PATTERN = re.compile(r"(未来方向|future work|后续工作)", re.IGNORECASE)
DECISION_PATTERN = re.compile(r"(建议投入|暂缓投入|建议|决策)", re.IGNORECASE)
TITLE_CONTRIBUTION_EFFECT_PATTERN = re.compile(
    r"(：|:|-).*(提升|降低|加速|效果|场景|应用|准确率|成本|效率|性能|稳定)",
    re.IGNORECASE,
)
INNOVATION_PATTERN = re.compile(r"(创新|novel|新模块|关键模块)", re.IGNORECASE)
ARCHITECTURE_PATTERN = re.compile(r"(架构|pipeline|数据流|模块)", re.IGNORECASE)
LIST_LINE_PATTERN = re.compile(r"^\s*(?:[-*•]|[0-9]+\.)\s+")
INFORMAL_TERM_PATTERN = re.compile(
    r"(省流|油表|必读|真金白银|吊打|碾压|王炸|爆款|小白|闭眼冲|一图看懂)",
    re.IGNORECASE,
)
INTERNAL_EVIDENCE_ANCHOR_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*{0,2})?\s*(?:证据锚点|Evidence\s+anchors?)\s*(?:\*{0,2})?\s*[:：]",
    re.IGNORECASE,
)
MARKDOWN_H1_PATTERN = re.compile(r"^\s*#\s+(.+?)\s*$")
SUMMARY_FIRST_LAYER_PATTERN = re.compile(r"^\s*##\s*1\)")
ABSTRACT_HEADING_PATTERN = re.compile(r"^\s*#+\s*(abstract|摘要)\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(
    r"\barxiv\s*:\s*[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?\b", re.IGNORECASE
)
ACCEPTED_PATTERN = re.compile(
    r"\baccepted\s+(?:to|at|for)\s+([^.;\n]+)",
    re.IGNORECASE,
)
SUBMITTED_PATTERN = re.compile(
    r"\bsubmitted\s+to\s+([^.;\n]+)",
    re.IGNORECASE,
)
PREPRINT_PATTERN = re.compile(r"\bpreprint\b", re.IGNORECASE)
AUTHOR_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|，|、|;|；|\band\b|&|/|\|)\s*", re.IGNORECASE
)
AUTHOR_SUFFIX_PATTERN = re.compile(r"[0-9*†‡]+$")
AUTHOR_INLINE_MARKUP_PATTERN = re.compile(r"<sup>[^<]*</sup>", re.IGNORECASE)
AUTHOR_LATEX_TEXT_PATTERN = re.compile(r"(?:\\text|text|ext)\{[^{}]*\}", re.IGNORECASE)
AUTHOR_LATEX_SUP_PATTERN = re.compile(r"\$\s*\^\s*\{[^{}]*\}\s*\$?")
AUTHOR_CARET_SUP_PATTERN = re.compile(r"\^\{?\s*[0-9,\s*†‡]+\}?")
AUTHOR_DISPLAY_LATEX_SUP_PATTERN = re.compile(r"\$\s*\^\s*\{([^{}]*)\}\s*\$?")
AUTHOR_DISPLAY_CARET_SUP_PATTERN = re.compile(r"\^\s*\{([^{}]*)\}")
AUTHOR_ALLOWED_SUPERSCRIPT_CHARS_PATTERN = re.compile(r"[^0-9,\s*†‡]")
AUTHOR_NAME_PATTERN = re.compile(r"[A-Z][a-zA-Z'`.-]+\s+[A-Z][a-zA-Z'`.-]+")
AFFILIATION_KEYWORD_PATTERN = re.compile(
    r"(university|institute|college|school|department|lab|laboratory|center|centre|"
    r"academy|hospital|research|corp\.?|inc\.?|ltd\.?|公司|大学|学院|研究院|实验室|研究所|中心)",
    re.IGNORECASE,
)
NON_METADATA_LINE_PATTERN = re.compile(
    r"(arxiv|doi|http|abstract|摘要|figure|table)",
    re.IGNORECASE,
)
UNKNOWN_EVIDENCE = "[未在原文找到直接证据]"
TITLECASE_SMALL_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "nor",
    "on",
    "at",
    "to",
    "from",
    "by",
    "of",
    "in",
    "via",
    "with",
    "as",
    "per",
}

MIN_FIGURE_TABLE_REF_COUNT = 2
MAX_LIST_LINE_RATIO = 0.35


class PaperSummarizationPipeline:
    """Coordinates MinerU parsing and OpenAI summarization."""

    def __init__(
        self,
        parser: MinerUPdfParser,
        summarizer: OpenAISummarizer,
        output_dir: Path,
        evidence_enricher: SummaryEvidenceEnricher | None = None,
        summary_format: str = "five_layers_v1",
        progress_tracker: "RichProgressTracker | None" = None,
    ) -> None:
        self.parser = parser
        self.summarizer = summarizer
        self.output_dir = output_dir
        self.evidence_enricher = evidence_enricher or SummaryEvidenceEnricher()
        self.summary_format = summary_format
        self.progress_tracker = progress_tracker

    def run(
        self,
        input_path: Path,
        max_files: int | None = None,
        skip_existing: bool = True,
    ) -> list[PipelineResult]:
        pdf_paths = discover_pdf_paths(input_path=input_path, max_files=max_files)
        results: list[PipelineResult] = []

        for pdf_path in pdf_paths:
            result = self.process_one(
                pdf_path=pdf_path,
                input_root=input_path,
                skip_existing=skip_existing,
            )
            results.append(result)

        return results

    def run_urls(
        self,
        pdf_urls: list[str],
        skip_existing: bool = True,
    ) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for pdf_url in pdf_urls:
            result = self.process_one_url(
                pdf_url=pdf_url,
                skip_existing=skip_existing,
            )
            results.append(result)
        return results

    def process_one(
        self,
        pdf_path: Path,
        input_root: Path,
        skip_existing: bool,
    ) -> PipelineResult:
        """Process a single PDF file with optional progress tracking."""
        if self.progress_tracker:
            from .progress import track_processing
            with track_processing(pdf_path.name, self.progress_tracker) as stage:
                return self._process_one_with_stage(
                    pdf_path=pdf_path,
                    input_root=input_root,
                    skip_existing=skip_existing,
                    stage=stage,
                )
        else:
            return self._process_one_legacy(
                pdf_path=pdf_path,
                input_root=input_root,
                skip_existing=skip_existing,
            )

    def _process_one_legacy(
        self,
        pdf_path: Path,
        input_root: Path,
        skip_existing: bool,
    ) -> PipelineResult:
        """Legacy processing without progress tracking."""
        parsed_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="parsed_markdown",
            new_suffix=".md",
        )
        summary_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="summaries",
            new_suffix=".md",
        )
        metadata_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="metadata",
            new_suffix=".json",
        )

        if skip_existing and summary_path.exists():
            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path,
                success=True,
                error=None,
            )

        try:
            artifact_dir = build_artifact_dir(
                pdf_path,
                input_root=input_root,
                output_root=self.output_dir,
            )
            parsed_paper = self.parser.parse_pdf(
                pdf_path=pdf_path, artifact_dir=artifact_dir
            )
            parsed_path.write_text(parsed_paper.markdown_text, encoding="utf-8")
            paper_metadata = self._extract_paper_metadata(
                paper_text=parsed_paper.markdown_text,
                fallback_title=pdf_path.stem,
            )

            summary_text, summary_token_usage = self._summarize_with_optional_metrics(
                paper_title=paper_metadata["title"],
                paper_text=parsed_paper.markdown_text,
            )
            summary_text, summary_evidence_coverage = (
                self.evidence_enricher.enrich_summary(
                    summary_text=summary_text,
                    artifact_dir=artifact_dir,
                    summary_path=summary_path,
                )
            )
            summary_text = self._strip_internal_scaffolding(summary_text)
            summary_text = self._inject_metadata_header(
                summary_text=summary_text,
                paper_metadata=paper_metadata,
            )
            summary_text = self._improve_readability(summary_text)
            summary_path.write_text(summary_text, encoding="utf-8")

            summary_coverage = self._build_summary_coverage(summary_text=summary_text)
            summary_style_coverage = self._build_summary_style_coverage(
                summary_text=summary_text
            )

            metadata = {
                "pdf_path": str(pdf_path),
                "parsed_markdown_path": str(parsed_path),
                "summary_path": str(summary_path),
                "artifact_dir": str(artifact_dir),
                "summary_format_version": self.summary_format,
                "parsed_chars": len(parsed_paper.markdown_text),
                "summary_chars": len(summary_text),
                "summary_coverage": summary_coverage,
                "summary_style_coverage": summary_style_coverage,
                "summary_evidence_coverage": summary_evidence_coverage,
                "summary_token_usage": summary_token_usage,
                "paper_metadata": paper_metadata,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path,
                summary_path=summary_path,
                success=True,
                error=None,
            )
        except Exception as exc:
            error_metadata = {
                "pdf_path": str(pdf_path),
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(error_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path if summary_path.exists() else None,
                success=False,
                error=str(exc),
            )

    def _process_one_with_stage(
        self,
        pdf_path: Path,
        input_root: Path,
        skip_existing: bool,
        stage: "StageUpdater",
    ) -> PipelineResult:
        """Process with Rich progress tracking."""
        parsed_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="parsed_markdown",
            new_suffix=".md",
        )
        summary_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="summaries",
            new_suffix=".md",
        )
        metadata_path = build_output_path(
            pdf_path=pdf_path,
            input_root=input_root,
            output_root=self.output_dir,
            category="metadata",
            new_suffix=".json",
        )

        if skip_existing and summary_path.exists():
            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path,
                success=True,
                error=None,
            )

        try:
            artifact_dir = build_artifact_dir(
                pdf_path,
                input_root=input_root,
                output_root=self.output_dir,
            )

            # Step 1: MinerU Upload
            file_size_kb = pdf_path.stat().st_size / 1024
            stage.start("MinerU Upload", f"{file_size_kb:.1f} KB")
            # Upload happens during parse_pdf, so we mark it complete after
            # Note: actual upload is in parse_pdf, we'll update details there

            # Step 2: MinerU Parsing
            stage.start("MinerU Parsing", "Waiting for MinerU...")

            # Create a progress callback for parsing
            def _parse_progress(step: str, details: str) -> None:
                if step == "parsing":
                    stage.update_details("MinerU Parsing", details)

            # Temporarily set the callback on parser
            original_callback = getattr(self.parser, 'progress_callback', None)
            self.parser.progress_callback = _parse_progress

            try:
                parsed_paper = self.parser.parse_pdf(
                    pdf_path=pdf_path, artifact_dir=artifact_dir
                )
            finally:
                self.parser.progress_callback = original_callback

            stage.complete("MinerU Parsing", f"{len(parsed_paper.markdown_text):,} chars")
            stage.complete("MinerU Upload", "✓")  # Complete upload stage too

            parsed_path.write_text(parsed_paper.markdown_text, encoding="utf-8")

            # Step 3: Extract metadata
            paper_metadata = self._extract_paper_metadata(
                paper_text=parsed_paper.markdown_text,
                fallback_title=pdf_path.stem,
            )

            # Step 4: Story Planning
            stage.start("Story Planning", "Generating narrative plan...")

            # Step 5: Layer Generation (1-5)
            layer_stages = [
                "Layer 1/5 (TL;DR)",
                "Layer 2/5 (Motivation)",
                "Layer 3/5 (Method)",
                "Layer 4/5 (Results)",
                "Layer 5/5 (Insights)",
            ]

            def _layer_progress(layer_key: str, details: str) -> None:
                # Map layer_key to stage name
                layer_map = {
                    "story_planning": "Story Planning",
                    "layer_1": "Layer 1/5 (TL;DR)",
                    "layer_2": "Layer 2/5 (Motivation)",
                    "layer_3": "Layer 3/5 (Method)",
                    "layer_4": "Layer 4/5 (Results)",
                    "layer_5": "Layer 5/5 (Insights)",
                    "review": "Review",
                    "rewrite": "Final Rewrite",
                }
                stage_name = layer_map.get(layer_key)
                if stage_name:
                    if "✓" in details or "Generating" not in details:
                        # Completed
                        stage.complete(stage_name, details.replace("✓ ", ""))
                    else:
                        stage.start(stage_name, details.replace("Generating ", ""))

            # Temporarily set callback on summarizer
            original_summarizer_callback = getattr(self.summarizer, 'progress_callback', None)
            self.summarizer.progress_callback = _layer_progress

            try:
                summary_text, summary_token_usage = self._summarize_with_optional_metrics(
                    paper_title=paper_metadata["title"],
                    paper_text=parsed_paper.markdown_text,
                )
            finally:
                self.summarizer.progress_callback = original_summarizer_callback

            # Mark any remaining layer stages as complete
            for layer_stage in layer_stages:
                # Check if already completed
                file_idx = stage.file_idx
                file_prog = self.progress_tracker.files[file_idx]
                for s in file_prog.stages:
                    if s.name == layer_stage and s.status != "completed":
                        stage.complete(layer_stage)

            # Step 6: Review (if enabled)
            if hasattr(self.summarizer, 'review_enabled') and self.summarizer.review_enabled:
                # Already handled by callback
                pass
            else:
                stage.complete("Review", "Skipped")

            # Step 7: Final Rewrite (if enabled)
            if hasattr(self.summarizer, 'rewrite_enabled') and self.summarizer.rewrite_enabled:
                # Already handled by callback
                pass
            else:
                stage.complete("Final Rewrite", "Skipped")

            # Update token usage in tracker
            total_tokens = 0
            if isinstance(summary_token_usage, dict):
                total_tokens = summary_token_usage.get("aggregate", {}).get("total_tokens", 0)
                # Estimate cost at $0.002 per 1K tokens (rough estimate)
                estimated_cost = total_tokens * 0.002 / 1000
                self.progress_tracker.update_token_usage(stage.file_idx, total_tokens, estimated_cost)

            # Step 8: Enrich summary
            summary_text, summary_evidence_coverage = (
                self.evidence_enricher.enrich_summary(
                    summary_text=summary_text,
                    artifact_dir=artifact_dir,
                    summary_path=summary_path,
                )
            )

            # Step 9: Post-process and save
            stage.start("Save Results", "Finalizing...")
            summary_text = self._strip_internal_scaffolding(summary_text)
            summary_text = self._inject_metadata_header(
                summary_text=summary_text,
                paper_metadata=paper_metadata,
            )
            summary_text = self._improve_readability(summary_text)
            summary_path.write_text(summary_text, encoding="utf-8")
            stage.complete("Save Results", f"{len(summary_text):,} chars")

            summary_coverage = self._build_summary_coverage(summary_text=summary_text)
            summary_style_coverage = self._build_summary_style_coverage(
                summary_text=summary_text
            )

            metadata = {
                "pdf_path": str(pdf_path),
                "parsed_markdown_path": str(parsed_path),
                "summary_path": str(summary_path),
                "artifact_dir": str(artifact_dir),
                "summary_format_version": self.summary_format,
                "parsed_chars": len(parsed_paper.markdown_text),
                "summary_chars": len(summary_text),
                "summary_coverage": summary_coverage,
                "summary_style_coverage": summary_style_coverage,
                "summary_evidence_coverage": summary_evidence_coverage,
                "summary_token_usage": summary_token_usage,
                "paper_metadata": paper_metadata,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path,
                summary_path=summary_path,
                success=True,
                error=None,
            )
        except Exception as exc:
            # Mark current stage as failed
            stage.fail("MinerU Parsing" if "MinerU" in str(exc) else "Final Rewrite", str(exc))

            error_metadata = {
                "pdf_path": str(pdf_path),
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(error_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return PipelineResult(
                pdf_path=pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path if summary_path.exists() else None,
                success=False,
                error=str(exc),
            )

    def process_one_url(self, pdf_url: str, skip_existing: bool) -> PipelineResult:
        pdf_stub = self._stub_from_url(pdf_url)
        parsed_path = self.output_dir / "parsed_markdown" / f"{pdf_stub}.md"
        summary_path = self.output_dir / "summaries" / f"{pdf_stub}.md"
        metadata_path = self.output_dir / "metadata" / f"{pdf_stub}.json"
        artifact_dir = self.output_dir / "mineru_artifacts" / pdf_stub

        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        logical_pdf_path = Path(f"{pdf_stub}.pdf")
        if skip_existing and summary_path.exists():
            return PipelineResult(
                pdf_path=logical_pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path,
                success=True,
                error=None,
            )

        try:
            parsed_paper = self.parser.parse_pdf_url(
                pdf_url=pdf_url,
                artifact_dir=artifact_dir,
                file_name=f"{pdf_stub}.pdf",
            )
            parsed_path.write_text(parsed_paper.markdown_text, encoding="utf-8")
            paper_metadata = self._extract_paper_metadata(
                paper_text=parsed_paper.markdown_text,
                fallback_title=pdf_stub,
            )

            summary_text, summary_token_usage = self._summarize_with_optional_metrics(
                paper_title=paper_metadata["title"],
                paper_text=parsed_paper.markdown_text,
            )
            summary_text, summary_evidence_coverage = (
                self.evidence_enricher.enrich_summary(
                    summary_text=summary_text,
                    artifact_dir=artifact_dir,
                    summary_path=summary_path,
                )
            )
            summary_text = self._strip_internal_scaffolding(summary_text)
            summary_text = self._inject_metadata_header(
                summary_text=summary_text,
                paper_metadata=paper_metadata,
            )
            summary_text = self._improve_readability(summary_text)
            summary_path.write_text(summary_text, encoding="utf-8")

            summary_coverage = self._build_summary_coverage(summary_text=summary_text)
            summary_style_coverage = self._build_summary_style_coverage(
                summary_text=summary_text
            )

            metadata = {
                "pdf_url": pdf_url,
                "parsed_markdown_path": str(parsed_path),
                "summary_path": str(summary_path),
                "artifact_dir": str(artifact_dir),
                "summary_format_version": self.summary_format,
                "parsed_chars": len(parsed_paper.markdown_text),
                "summary_chars": len(summary_text),
                "summary_coverage": summary_coverage,
                "summary_style_coverage": summary_style_coverage,
                "summary_evidence_coverage": summary_evidence_coverage,
                "summary_token_usage": summary_token_usage,
                "paper_metadata": paper_metadata,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return PipelineResult(
                pdf_path=logical_pdf_path,
                parsed_markdown_path=parsed_path,
                summary_path=summary_path,
                success=True,
                error=None,
            )
        except Exception as exc:
            error_metadata = {
                "pdf_url": pdf_url,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(
                json.dumps(error_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return PipelineResult(
                pdf_path=logical_pdf_path,
                parsed_markdown_path=parsed_path if parsed_path.exists() else None,
                summary_path=summary_path if summary_path.exists() else None,
                success=False,
                error=str(exc),
            )

    def _summarize_with_optional_metrics(
        self,
        paper_title: str,
        paper_text: str,
    ) -> tuple[str, dict[str, object]]:
        if hasattr(self.summarizer, "summarize_with_metrics"):
            result = self.summarizer.summarize_with_metrics(
                paper_title=paper_title,
                paper_text=paper_text,
            )
            return result.summary_text, result.token_usage.to_dict()

        summary_text = self.summarizer.summarize(
            paper_title=paper_title,
            paper_text=paper_text,
        )
        return summary_text, {
            "enabled": False,
            "usage_available": False,
            "steps": [],
            "aggregate": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "step_count": 0,
            },
            "by_phase": {
                "planner": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "step_count": 0,
                },
                "layer_generation": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "step_count": 0,
                },
                "review": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "step_count": 0,
                },
                "rewrite": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "step_count": 0,
                },
            },
        }

    def _extract_paper_metadata(
        self,
        paper_text: str,
        fallback_title: str,
    ) -> dict[str, object]:
        lines = [line.strip() for line in paper_text.splitlines()]
        title, title_line_index = self._extract_title(
            lines=lines, fallback_title=fallback_title
        )
        authors, authors_display = self._extract_authors(
            lines=lines,
            title_line_index=title_line_index,
        )
        affiliations = self._extract_affiliations(
            lines=lines,
            title_line_index=title_line_index,
        )
        publication = self._extract_publication(lines=lines)
        return {
            "title": title,
            "authors": authors,
            "authors_display": authors_display,
            "affiliations": affiliations,
            "publication_platform": publication["platform"],
            "publication_status": publication["status"],
        }

    def _extract_title(self, lines: list[str], fallback_title: str) -> tuple[str, int]:
        for index, line in enumerate(lines[:80]):
            heading_match = MARKDOWN_H1_PATTERN.match(line)
            if not heading_match:
                continue
            candidate = self._clean_inline_metadata(heading_match.group(1))
            if candidate and candidate.lower() not in {"title", "paper"}:
                return self._format_title_for_display(candidate), index

        for index, line in enumerate(lines[:40]):
            candidate = self._clean_inline_metadata(line.lstrip("# "))
            if not candidate:
                continue
            if ABSTRACT_HEADING_PATTERN.match(line):
                break
            if NON_METADATA_LINE_PATTERN.search(candidate):
                continue
            if len(candidate) > 180:
                continue
            return self._format_title_for_display(candidate), index

        return self._format_title_for_display(fallback_title), -1

    def _format_title_for_display(self, title: str) -> str:
        if not self._is_mostly_uppercase_title(title):
            return title

        tokens = title.split()
        formatted: list[str] = []
        force_capitalize = True
        last_index = len(tokens) - 1

        for index, token in enumerate(tokens):
            formatted.append(
                self._format_title_token(
                    token=token,
                    force_capitalize=force_capitalize
                    or index == 0
                    or index == last_index,
                )
            )
            force_capitalize = token.endswith(":")

        return " ".join(formatted)

    def _is_mostly_uppercase_title(self, title: str) -> bool:
        letters = [ch for ch in title if ch.isalpha()]
        if len(letters) < 8:
            return False
        uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        return uppercase_ratio >= 0.75

    def _format_title_token(self, token: str, force_capitalize: bool) -> str:
        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9'`.-]+)([^A-Za-z0-9]*)$", token)
        if not match:
            return token

        prefix, core, suffix = match.groups()
        if (
            core.isupper()
            and core.isalpha()
            and len(core) <= 5
            and core.lower() not in TITLECASE_SMALL_WORDS
        ):
            return f"{prefix}{core}{suffix}"

        lowered = core.lower()
        if not force_capitalize and lowered in TITLECASE_SMALL_WORDS:
            return f"{prefix}{lowered}{suffix}"

        if "-" in lowered:
            parts = lowered.split("-")
            core_text = "-".join(
                part[:1].upper() + part[1:] if part else part for part in parts
            )
        else:
            core_text = lowered[:1].upper() + lowered[1:]
        return f"{prefix}{core_text}{suffix}"

    def _extract_authors(
        self,
        lines: list[str],
        title_line_index: int,
    ) -> tuple[list[str], str | None]:
        start_index = 0 if title_line_index < 0 else title_line_index + 1
        for raw_line in lines[start_index : start_index + 12]:
            candidate_line = self._normalize_author_line(raw_line)
            display_line = self._normalize_author_display_line(raw_line)
            if not candidate_line:
                continue
            if candidate_line.startswith("#"):
                continue
            if NON_METADATA_LINE_PATTERN.search(candidate_line):
                continue
            if AFFILIATION_KEYWORD_PATTERN.search(candidate_line):
                continue

            authors = self._parse_authors_from_line(candidate_line)
            if len(authors) >= 2:
                return list(dict.fromkeys(authors))[:10], display_line

        return [], None

    def _normalize_author_line(self, raw_line: str) -> str:
        normalized = self._clean_inline_metadata(raw_line)
        normalized = AUTHOR_INLINE_MARKUP_PATTERN.sub(" ", normalized)
        normalized = AUTHOR_LATEX_TEXT_PATTERN.sub("", normalized)
        normalized = AUTHOR_LATEX_SUP_PATTERN.sub(" ", normalized)
        normalized = AUTHOR_CARET_SUP_PATTERN.sub(" ", normalized)
        normalized = normalized.replace("{", " ").replace("}", " ")
        normalized = normalized.replace("$", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _normalize_author_display_line(self, raw_line: str) -> str:
        normalized = self._clean_inline_metadata(raw_line)
        normalized = AUTHOR_INLINE_MARKUP_PATTERN.sub(" ", normalized)
        normalized = AUTHOR_LATEX_TEXT_PATTERN.sub("", normalized)
        normalized = AUTHOR_DISPLAY_LATEX_SUP_PATTERN.sub(
            self._sanitize_latex_superscript_match, normalized
        )
        normalized = AUTHOR_DISPLAY_CARET_SUP_PATTERN.sub(
            self._sanitize_caret_superscript_match, normalized
        )
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _sanitize_latex_superscript_match(self, match: re.Match[str]) -> str:
        cleaned = self._sanitize_superscript_content(match.group(1))
        if not cleaned:
            return ""
        return f"$^{{{cleaned}}}$"

    def _sanitize_caret_superscript_match(self, match: re.Match[str]) -> str:
        cleaned = self._sanitize_superscript_content(match.group(1))
        if not cleaned:
            return ""
        return f"^{{{cleaned}}}"

    def _sanitize_superscript_content(self, raw_content: str) -> str:
        cleaned = AUTHOR_ALLOWED_SUPERSCRIPT_CHARS_PATTERN.sub("", raw_content)
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    def _parse_authors_from_line(self, candidate_line: str) -> list[str]:
        candidates = [
            AUTHOR_SUFFIX_PATTERN.sub("", token).strip(" *†‡")
            for token in AUTHOR_SPLIT_PATTERN.split(candidate_line)
            if token.strip()
        ]
        authors = [
            token
            for token in candidates
            if len(token) >= 2
            and not AFFILIATION_KEYWORD_PATTERN.search(token)
            and not NON_METADATA_LINE_PATTERN.search(token)
        ]

        if len(authors) >= 2:
            return authors

        name_matches = [
            match.group(0).strip()
            for match in AUTHOR_NAME_PATTERN.finditer(candidate_line)
        ]
        return [
            name
            for name in name_matches
            if not AFFILIATION_KEYWORD_PATTERN.search(name)
            and not NON_METADATA_LINE_PATTERN.search(name)
        ]

    def _extract_affiliations(
        self, lines: list[str], title_line_index: int
    ) -> list[str]:
        start_index = 0 if title_line_index < 0 else title_line_index + 1
        affiliations: list[str] = []
        for raw_line in lines[start_index : start_index + 30]:
            candidate_line = self._clean_inline_metadata(raw_line)
            if not candidate_line:
                continue
            if candidate_line.startswith("#"):
                continue
            if NON_METADATA_LINE_PATTERN.search(candidate_line):
                continue
            if not AFFILIATION_KEYWORD_PATTERN.search(candidate_line):
                continue
            if candidate_line not in affiliations:
                affiliations.append(candidate_line)
            if len(affiliations) >= 3:
                break
        return affiliations

    def _extract_publication(self, lines: list[str]) -> dict[str, str]:
        head_text = "\n".join(lines[:120])

        accepted_match = ACCEPTED_PATTERN.search(head_text)
        if accepted_match:
            venue = self._clean_inline_metadata(accepted_match.group(1))
            return {
                "platform": venue or UNKNOWN_EVIDENCE,
                "status": "accepted",
            }

        submitted_match = SUBMITTED_PATTERN.search(head_text)
        if submitted_match:
            venue = self._clean_inline_metadata(submitted_match.group(1))
            return {
                "platform": venue or UNKNOWN_EVIDENCE,
                "status": "submitted",
            }

        if ARXIV_PATTERN.search(head_text):
            return {
                "platform": "arXiv",
                "status": "preprint",
            }

        if PREPRINT_PATTERN.search(head_text):
            return {
                "platform": UNKNOWN_EVIDENCE,
                "status": "preprint",
            }

        return {
            "platform": UNKNOWN_EVIDENCE,
            "status": UNKNOWN_EVIDENCE,
        }

    def _clean_inline_metadata(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.strip("|>\t ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _inject_metadata_header(
        self,
        summary_text: str,
        paper_metadata: dict[str, object],
    ) -> str:
        lines = summary_text.splitlines()
        title_index = next(
            (
                index
                for index, line in enumerate(lines)
                if TITLE_LINE_PATTERN.match(line)
            ),
            -1,
        )

        if title_index < 0:
            lines = [f"# {paper_metadata['title']}", "", *lines]
            title_index = 0

        layer_index = next(
            (
                index
                for index, line in enumerate(
                    lines[title_index + 1 :], start=title_index + 1
                )
                if SUMMARY_FIRST_LAYER_PATTERN.match(line)
            ),
            len(lines),
        )

        authors = paper_metadata.get("authors")
        authors_display = paper_metadata.get("authors_display")
        affiliations = paper_metadata.get("affiliations")
        if isinstance(authors_display, str) and authors_display.strip():
            authors_text = authors_display.strip()
        else:
            authors_text = (
                ", ".join(authors)
                if isinstance(authors, list) and authors
                else UNKNOWN_EVIDENCE
            )
        affiliations_text = (
            "；".join(affiliations)
            if isinstance(affiliations, list) and affiliations
            else UNKNOWN_EVIDENCE
        )

        platform = str(paper_metadata.get("publication_platform") or UNKNOWN_EVIDENCE)
        status = str(paper_metadata.get("publication_status") or UNKNOWN_EVIDENCE)

        header_block = [
            f"> 标题：{paper_metadata['title']}",
            f"> 作者：{authors_text}",
            f"> 单位：{affiliations_text}",
        ]

        if platform != UNKNOWN_EVIDENCE:
            if status != UNKNOWN_EVIDENCE:
                header_block.append(f"> 发布信息：{platform}（{status}）")
            else:
                header_block.append(f"> 发布信息：{platform}")

        rebuilt = [
            *lines[: title_index + 1],
            "",
            *header_block,
            "",
            *lines[layer_index:],
        ]
        return "\n".join(rebuilt).rstrip() + "\n"

    def _improve_readability(self, summary_text: str) -> str:
        improved_lines: list[str] = []
        split_pattern = re.compile(r"^(.+?[。！？!?])\s*(.+)$")
        one_liner_inline_pattern = re.compile(
            r"^(?:一句话总结|One-?Liner|TL;DR|省流)\s*[:：]\s*(.+)$",
            re.IGNORECASE,
        )
        expansion_inline_pattern = re.compile(
            r"^(?:一页卡片(?:（易懂短版）?)?|展开来讲)\s*[:：]\s*(.*)$"
        )

        for line in summary_text.splitlines():
            stripped = line.strip()

            one_liner_match = one_liner_inline_pattern.match(stripped)
            if one_liner_match and one_liner_match.group(1).strip():
                content = one_liner_match.group(1).strip()
                improved_lines.append("一句话总结：")
                improved_lines.append("")
                improved_lines.append("> [!TIP]")
                improved_lines.append(f"> {content}")
                continue

            expansion_match = expansion_inline_pattern.match(stripped)
            if expansion_match:
                content = expansion_match.group(1).strip()
                improved_lines.append("展开来讲：")
                if content:
                    improved_lines.append("")
                    improved_lines.append(content)
                continue

            if not self._should_split_paragraph(stripped):
                improved_lines.append(line)
                continue

            split_match = split_pattern.match(stripped)
            if not split_match:
                improved_lines.append(line)
                continue

            first_sentence = split_match.group(1).strip()
            remaining = split_match.group(2).strip()
            if not remaining:
                improved_lines.append(line)
                continue

            if not (first_sentence.startswith("**") and first_sentence.endswith("**")):
                first_sentence = f"**{first_sentence}**"

            improved_lines.append(first_sentence)
            improved_lines.append("")
            improved_lines.append(remaining)

        return "\n".join(improved_lines).rstrip() + "\n"

    def _should_split_paragraph(self, stripped_line: str) -> bool:
        if not stripped_line:
            return False
        if len(stripped_line) < 90:
            return False
        if stripped_line.startswith(("#", ">", "-", "*", "|", "![", "```")):
            return False
        if re.match(r"^[0-9]+\.\s", stripped_line):
            return False
        if "。" not in stripped_line and not re.search(r"[.!?]", stripped_line):
            return False
        return True

    def _stub_from_url(self, pdf_url: str) -> str:
        parsed = urlparse(pdf_url)
        candidate = Path(parsed.path).name or "remote_pdf"
        stem = Path(candidate).stem or "remote_pdf"
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)

    def _build_summary_coverage(self, summary_text: str) -> dict[str, object]:
        present_layers: list[str] = []
        missing_layers: list[str] = []
        for layer_key, heading in SUMMARY_LAYER_HEADINGS.items():
            if heading in summary_text:
                present_layers.append(layer_key)
            else:
                missing_layers.append(layer_key)

        separator_count = len(SEPARATOR_LINE_PATTERN.findall(summary_text))
        total_layers = len(SUMMARY_LAYER_HEADINGS)
        coverage_ratio = (
            1.0 if total_layers == 0 else round(len(present_layers) / total_layers, 3)
        )

        return {
            "required_layer_headings": SUMMARY_LAYER_HEADINGS,
            "present_layers": present_layers,
            "missing_layers": missing_layers,
            "separator_count": separator_count,
            "required_separator_count": 4,
            "coverage_ratio": coverage_ratio,
            "is_complete": len(missing_layers) == 0 and separator_count >= 4,
        }

    def _build_summary_style_coverage(self, summary_text: str) -> dict[str, object]:
        layer_bodies = self._extract_layer_bodies(summary_text=summary_text)

        content_lines = [
            line.strip() for line in summary_text.splitlines() if line.strip()
        ]
        list_line_count = len(
            [line for line in content_lines if LIST_LINE_PATTERN.match(line)]
        )
        list_line_ratio = (
            0.0 if not content_lines else round(list_line_count / len(content_lines), 3)
        )
        figure_table_ref_count = len(FIGURE_TABLE_REF_PATTERN.findall(summary_text))
        informal_term_hits = sorted(set(INFORMAL_TERM_PATTERN.findall(summary_text)))

        title_match = TITLE_LINE_PATTERN.search(summary_text)
        has_title_line = title_match is not None
        title_line = title_match.group(0).strip() if title_match else ""
        has_original_title_line = bool(ORIGINAL_TITLE_LINE_PATTERN.search(summary_text))
        has_title_contribution_effect_pattern = bool(
            TITLE_CONTRIBUTION_EFFECT_PATTERN.search(title_line)
        )

        layer_1 = layer_bodies.get("hook_tldr", "")
        layer_2 = layer_bodies.get("motivation_gap", "")
        layer_3 = layer_bodies.get("method_mechanism", "")
        layer_4 = layer_bodies.get("proof_results", "")
        layer_5 = layer_bodies.get("insights_decision", "")

        has_one_liner = bool(ONE_LINER_PATTERN.search(layer_1))
        has_pain_point = bool(PAIN_POINT_PATTERN.search(layer_2))
        has_research_gap = bool(GAP_PATTERN.search(layer_2))
        has_key_insight = bool(INSIGHT_PATTERN.search(layer_2))
        has_architecture_ref = bool(FIGURE_TABLE_REF_PATTERN.search(layer_3))
        has_architecture_explain = bool(ARCHITECTURE_PATTERN.search(layer_3))
        innovation_marker_count = len(INNOVATION_PATTERN.findall(layer_3))
        has_sota_statement = bool(SOTA_PATTERN.search(layer_4))
        has_ablation_statement = bool(ABLATION_PATTERN.search(layer_4))
        has_visual_case = bool(VISUAL_CASE_PATTERN.search(layer_4))
        limitations_count = len(LIMITATION_PATTERN.findall(layer_5))
        has_future_work = bool(FUTURE_WORK_PATTERN.search(layer_5))
        has_decision = bool(DECISION_PATTERN.search(layer_5))

        missing_requirements: list[str] = []
        if not has_title_line:
            missing_requirements.append("missing_title_line")
        if not has_original_title_line:
            missing_requirements.append("missing_original_title_line")
        if not has_title_contribution_effect_pattern:
            missing_requirements.append("missing_title_contribution_effect_pattern")
        if not has_one_liner:
            missing_requirements.append("missing_one_liner")
        if not has_pain_point:
            missing_requirements.append("missing_pain_point")
        if not has_research_gap:
            missing_requirements.append("missing_research_gap")
        if not has_key_insight:
            missing_requirements.append("missing_key_insight")
        if not has_architecture_ref:
            missing_requirements.append("missing_architecture_ref")
        if not has_architecture_explain:
            missing_requirements.append("missing_architecture_explain")
        if innovation_marker_count < 1:
            missing_requirements.append("missing_innovation_points")
        if not has_sota_statement:
            missing_requirements.append("missing_sota_statement")
        if not has_ablation_statement:
            missing_requirements.append("missing_ablation_statement")
        if not has_visual_case:
            missing_requirements.append("missing_visual_case")
        if limitations_count < 1:
            missing_requirements.append("missing_limitations")
        if not has_future_work:
            missing_requirements.append("missing_future_work")
        if figure_table_ref_count < MIN_FIGURE_TABLE_REF_COUNT:
            missing_requirements.append("insufficient_figure_table_refs")
        if list_line_ratio > MAX_LIST_LINE_RATIO:
            missing_requirements.append("too_many_list_lines")
        if informal_term_hits:
            missing_requirements.append("contains_informal_terms")

        return {
            "has_title_line": has_title_line,
            "has_original_title_line": has_original_title_line,
            "has_title_contribution_effect_pattern": has_title_contribution_effect_pattern,
            "has_one_liner": has_one_liner,
            "has_pain_point": has_pain_point,
            "has_research_gap": has_research_gap,
            "has_key_insight": has_key_insight,
            "has_architecture_ref": has_architecture_ref,
            "has_architecture_explain": has_architecture_explain,
            "innovation_marker_count": innovation_marker_count,
            "has_sota_statement": has_sota_statement,
            "has_ablation_statement": has_ablation_statement,
            "has_visual_case": has_visual_case,
            "limitations_count": limitations_count,
            "has_future_work": has_future_work,
            "has_decision": has_decision,
            "figure_table_ref_count": figure_table_ref_count,
            "min_figure_table_ref_count": MIN_FIGURE_TABLE_REF_COUNT,
            "list_line_count": list_line_count,
            "list_line_ratio": list_line_ratio,
            "max_list_line_ratio": MAX_LIST_LINE_RATIO,
            "informal_term_hits": informal_term_hits,
            "missing_requirements": missing_requirements,
            "is_complete": len(missing_requirements) == 0,
        }

    def _strip_internal_scaffolding(self, summary_text: str) -> str:
        kept_lines: list[str] = []
        for line in summary_text.splitlines():
            if INTERNAL_EVIDENCE_ANCHOR_LINE_PATTERN.match(line.strip()):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).rstrip() + "\n"

    def _extract_layer_bodies(self, summary_text: str) -> dict[str, str]:
        lines = summary_text.splitlines()
        heading_re = re.compile(r"^\s*##\s*([1-5])\)\s+(.+?)\s*$")

        layer_ranges: list[tuple[str, int]] = []
        for idx, line in enumerate(lines):
            match = heading_re.match(line)
            if not match:
                continue
            layer_key = LAYER_INDEX_TO_KEY.get(match.group(1))
            if layer_key is None:
                continue
            layer_ranges.append((layer_key, idx))

        layer_ranges.sort(key=lambda item: item[1])
        bodies: dict[str, str] = {key: "" for key in SUMMARY_LAYER_HEADINGS}

        for pos, (layer_key, start_idx) in enumerate(layer_ranges):
            end_idx = len(lines)
            if pos + 1 < len(layer_ranges):
                end_idx = layer_ranges[pos + 1][1]
            body_lines = lines[start_idx + 1 : end_idx]
            cleaned = "\n".join(
                line for line in body_lines if line.strip() != "---"
            ).strip()
            bodies[layer_key] = cleaned

        return bodies
