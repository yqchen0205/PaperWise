"""Typed models used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MinerUParseOptions:
    """Configuration sent to MinerU extraction APIs."""

    enable_formula: bool = True
    enable_table: bool = True
    language: str = "en"
    is_ocr: bool = True
    model_version: str = "vlm"


@dataclass(frozen=True)
class UploadFileSpec:
    """A local file to upload to MinerU before extraction."""

    path: Path
    data_id: str
    is_ocr: bool = True


@dataclass(frozen=True)
class UploadTarget:
    """Presigned upload target returned by MinerU."""

    name: str
    data_id: str
    upload_url: str


@dataclass(frozen=True)
class UploadBatch:
    """Batch upload metadata returned by MinerU."""

    batch_id: str
    targets: list[UploadTarget]


@dataclass(frozen=True)
class ExtractResultItem:
    """Single extraction result item from MinerU."""

    data_id: str
    file_name: str
    state: str
    result_url: str | None
    message: str | None


@dataclass(frozen=True)
class ParsedPaper:
    """Final parsed content from one PDF."""

    pdf_path: Path
    markdown_text: str
    artifact_dir: Path


@dataclass(frozen=True)
class PipelineResult:
    """Status for a single PDF processing run."""

    pdf_path: Path
    parsed_markdown_path: Path | None
    summary_path: Path | None
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class LLMCallUsage:
    """Token usage for one LLM call step."""

    step_name: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    usage_available: bool


@dataclass(frozen=True)
class SummarizationTokenUsage:
    """Token usage summary across all summarization steps."""

    enabled: bool
    usage_available: bool
    steps: list[LLMCallUsage]

    def to_dict(self) -> dict[str, object]:
        prompt_total = sum(step.prompt_tokens or 0 for step in self.steps)
        completion_total = sum(step.completion_tokens or 0 for step in self.steps)
        total_total = sum(step.total_tokens or 0 for step in self.steps)
        by_phase = {
            "planner": self._phase_totals(prefixes=("story_planner",)),
            "layer_generation": self._phase_totals(prefixes=("layer_",)),
            "review": self._phase_totals(prefixes=("review",)),
            "rewrite": self._phase_totals(prefixes=("rewrite", "final_polish")),
        }
        return {
            "enabled": self.enabled,
            "usage_available": self.usage_available,
            "steps": [
                {
                    "step_name": step.step_name,
                    "prompt_tokens": step.prompt_tokens,
                    "completion_tokens": step.completion_tokens,
                    "total_tokens": step.total_tokens,
                    "usage_available": step.usage_available,
                }
                for step in self.steps
            ],
            "aggregate": {
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
                "total_tokens": total_total,
                "step_count": len(self.steps),
            },
            "by_phase": by_phase,
        }

    def _phase_totals(self, prefixes: tuple[str, ...]) -> dict[str, int]:
        selected = [step for step in self.steps if step.step_name.startswith(prefixes)]
        return {
            "prompt_tokens": sum(step.prompt_tokens or 0 for step in selected),
            "completion_tokens": sum(step.completion_tokens or 0 for step in selected),
            "total_tokens": sum(step.total_tokens or 0 for step in selected),
            "step_count": len(selected),
        }


@dataclass(frozen=True)
class SummarizationResult:
    """Summarization payload including final text and usage trace."""

    summary_text: str
    token_usage: SummarizationTokenUsage
