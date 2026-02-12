"""Attach cited figures and tables to generated summaries."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIGURE_CAPTION_PATTERN = re.compile(r"(?:Figure|Fig\.?|图)\s*([0-9]+)", re.IGNORECASE)
TABLE_CAPTION_PATTERN = re.compile(r"(?:Table|表)\s*([0-9]+)", re.IGNORECASE)
SUBFIGURE_LABEL_PATTERN = re.compile(r"^\s*\(([a-z])\)", re.IGNORECASE)

FIGURE_REF_PATTERN = re.compile(
    r"(?:Figure|Fig\.?|图)\s*([0-9]+)\s*([a-z](?:\s*/\s*[a-z])*)?",
    re.IGNORECASE,
)
TABLE_REF_PATTERN = re.compile(r"(?:Table|表)\s*([0-9]+)", re.IGNORECASE)
EVIDENCE_META_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*{0,2})?\s*(?:证据锚点|证据|Evidence(?:\s+anchors?)?)\s*(?:\*{0,2})?\s*[:：]",
    re.IGNORECASE,
)
BLOCKQUOTE_LINE_PATTERN = re.compile(r"^\s*>")
RAW_TABLE_DATA_LINE_PATTERN = re.compile(r"^\s*Raw\s+table\s+data", re.IGNORECASE)
MARKDOWN_IMAGE_LINE_PATTERN = re.compile(r"^\s*!\[")
ATTACHMENT_HEADER_PATTERN = re.compile(
    r"^\s*>\s*Evidence\s+attachment:\s*(.+?)\s*$",
    re.IGNORECASE,
)
MARKDOWN_IMAGE_PATH_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

MAX_IMAGES_PER_FIGURE = 4
MAX_NEIGHBOR_DISTANCE = 12


@dataclass(frozen=True)
class _Reference:
    kind: str
    number: int
    sub_labels: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        if self.kind == "figure" and self.sub_labels:
            return f"{self.kind}:{self.number}:{'/'.join(self.sub_labels)}"
        return f"{self.kind}:{self.number}"

    @property
    def display_label(self) -> str:
        if self.kind == "figure" and self.sub_labels:
            return f"Figure {self.number}{'/'.join(self.sub_labels)}"
        if self.kind == "figure":
            return f"Figure {self.number}"
        return f"Table {self.number}"


@dataclass
class _ImageCandidate:
    item_index: int
    page_idx: int
    img_path: str
    caption_lines: list[str]
    figure_number: int | None
    sub_label: str | None


@dataclass
class _FigureAsset:
    number: int
    caption: str = ""
    image_paths: list[str] = field(default_factory=list)
    subfigure_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class _TableAsset:
    number: int
    caption: str = ""
    image_path: str | None = None
    table_body: str = ""


@dataclass
class _EvidenceIndex:
    content_list_path: str | None
    figures: dict[int, _FigureAsset]
    tables: dict[int, _TableAsset]


class SummaryEvidenceEnricher:
    """Inject nearby figure/table artifacts for cited references in summary markdown."""

    def enrich_summary(
        self,
        summary_text: str,
        artifact_dir: Path,
        summary_path: Path,
    ) -> tuple[str, dict[str, object]]:
        references = self._collect_unique_references(summary_text)
        if not references:
            return summary_text, self._build_coverage(
                content_list_path=None,
                references=[],
                attached_keys=set(),
                missing_keys=set(),
            )

        extract_dir = artifact_dir / "mineru_extract"
        evidence_index = self._build_evidence_index(extract_dir=extract_dir)
        existing_attachment_labels = self._collect_existing_attachment_labels(
            summary_text
        )
        existing_asset_paths = self._collect_existing_asset_paths(summary_text)

        output_lines: list[str] = []
        attached_keys: set[str] = set()
        missing_keys: set[str] = set()
        inserted_keys: set[str] = set()

        for line in summary_text.splitlines():
            output_lines.append(line)
            if self._is_evidence_meta_line(line):
                continue

            line_refs = self._extract_references_from_line(line)
            if not line_refs:
                continue

            for ref in line_refs:
                if ref.key in inserted_keys:
                    continue

                if ref.display_label in existing_attachment_labels:
                    attached_keys.add(ref.key)
                    inserted_keys.add(ref.key)
                    continue

                block_lines = self._build_attachment_block(
                    ref=ref,
                    evidence_index=evidence_index,
                    extract_dir=extract_dir,
                    summary_path=summary_path,
                )
                if block_lines:
                    block_asset_paths = self._collect_image_paths_from_lines(
                        block_lines
                    )
                    if block_asset_paths and block_asset_paths.issubset(
                        existing_asset_paths
                    ):
                        attached_keys.add(ref.key)
                        inserted_keys.add(ref.key)
                        continue

                    output_lines.append("")
                    output_lines.extend(block_lines)
                    output_lines.append("")
                    attached_keys.add(ref.key)
                    existing_asset_paths.update(block_asset_paths)
                else:
                    missing_keys.add(ref.key)

                inserted_keys.add(ref.key)

        enriched_text = "\n".join(output_lines).rstrip() + "\n"
        coverage = self._build_coverage(
            content_list_path=evidence_index.content_list_path,
            references=references,
            attached_keys=attached_keys,
            missing_keys=missing_keys,
        )
        return enriched_text, coverage

    def _build_evidence_index(self, extract_dir: Path) -> _EvidenceIndex:
        content_list_path = self._pick_content_list_path(extract_dir=extract_dir)
        if content_list_path is None:
            return _EvidenceIndex(content_list_path=None, figures={}, tables={})

        try:
            payload = json.loads(content_list_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _EvidenceIndex(
                content_list_path=str(content_list_path), figures={}, tables={}
            )

        items = self._collect_typed_items(payload)
        if not items:
            return _EvidenceIndex(
                content_list_path=str(content_list_path), figures={}, tables={}
            )

        image_candidates: list[_ImageCandidate] = []
        tables: dict[int, _TableAsset] = {}

        for index, item in enumerate(items):
            item_type = str(item.get("type", "")).lower()
            if item_type == "image":
                image_candidate = self._parse_image_candidate(
                    item=item, item_index=index
                )
                if image_candidate is not None:
                    image_candidates.append(image_candidate)
            elif item_type == "table":
                table_asset = self._parse_table_asset(item)
                if table_asset is not None:
                    tables[table_asset.number] = table_asset

        self._infer_missing_figure_numbers(image_candidates)

        figures: dict[int, _FigureAsset] = {}
        for candidate in image_candidates:
            if candidate.figure_number is None:
                continue

            figure = figures.setdefault(
                candidate.figure_number, _FigureAsset(number=candidate.figure_number)
            )
            if candidate.img_path and candidate.img_path not in figure.image_paths:
                figure.image_paths.append(candidate.img_path)

            if candidate.sub_label and candidate.img_path:
                figure.subfigure_paths.setdefault(
                    candidate.sub_label, candidate.img_path
                )

            caption_text = " ".join(candidate.caption_lines).strip()
            if caption_text and "figure" in caption_text.lower() and not figure.caption:
                figure.caption = caption_text

        for figure in figures.values():
            if not figure.subfigure_paths and figure.image_paths:
                for index, img_path in enumerate(
                    figure.image_paths[:MAX_IMAGES_PER_FIGURE]
                ):
                    label = chr(ord("a") + index)
                    figure.subfigure_paths[label] = img_path

        return _EvidenceIndex(
            content_list_path=str(content_list_path),
            figures=figures,
            tables=tables,
        )

    def _build_attachment_block(
        self,
        ref: _Reference,
        evidence_index: _EvidenceIndex,
        extract_dir: Path,
        summary_path: Path,
    ) -> list[str]:
        if ref.kind == "figure":
            return self._build_figure_block(
                ref=ref,
                figure=evidence_index.figures.get(ref.number),
                extract_dir=extract_dir,
                summary_path=summary_path,
            )

        return self._build_table_block(
            ref=ref,
            table=evidence_index.tables.get(ref.number),
            extract_dir=extract_dir,
            summary_path=summary_path,
        )

    def _build_figure_block(
        self,
        ref: _Reference,
        figure: _FigureAsset | None,
        extract_dir: Path,
        summary_path: Path,
    ) -> list[str]:
        if figure is None:
            return []

        selected: list[tuple[str, str]] = []
        missing_panels: list[str] = []

        if ref.sub_labels:
            for label in ref.sub_labels:
                img_path = figure.subfigure_paths.get(label)
                if img_path:
                    selected.append((label, img_path))
                else:
                    missing_panels.append(label)
        else:
            for img_path in figure.image_paths[:MAX_IMAGES_PER_FIGURE]:
                selected.append(("", img_path))

        resolved: list[tuple[str, str]] = []
        for panel_label, raw_path in selected:
            rel_path = self._resolve_relative_asset_path(
                extract_dir=extract_dir,
                raw_path=raw_path,
                summary_path=summary_path,
            )
            if rel_path:
                resolved.append((panel_label, rel_path))

        if not resolved:
            return []

        lines: list[str] = []

        for panel_label, rel_path in resolved:
            if panel_label:
                alt_text = f"Figure {ref.number}{panel_label}"
            else:
                alt_text = f"Figure {ref.number}"
            lines.append(f"![{alt_text}]({rel_path})")

        return lines

    def _build_table_block(
        self,
        ref: _Reference,
        table: _TableAsset | None,
        extract_dir: Path,
        summary_path: Path,
    ) -> list[str]:
        if table is None:
            return []

        lines: list[str] = []

        if table.image_path:
            rel_path = self._resolve_relative_asset_path(
                extract_dir=extract_dir,
                raw_path=table.image_path,
                summary_path=summary_path,
            )
            if rel_path:
                lines.append(f"![Table {ref.number}]({rel_path})")

        if table.table_body and not lines:
            lines.append("")
            lines.append("Raw table data (MinerU):")
            lines.append(table.table_body)

        return lines

    def _collect_unique_references(self, summary_text: str) -> list[_Reference]:
        references: list[_Reference] = []
        seen_keys: set[str] = set()

        for line in summary_text.splitlines():
            if self._is_evidence_meta_line(line):
                continue

            for ref in self._extract_references_from_line(line):
                if ref.key in seen_keys:
                    continue
                seen_keys.add(ref.key)
                references.append(ref)

        return references

    def _is_evidence_meta_line(self, line: str) -> bool:
        stripped = line.strip()
        return bool(
            EVIDENCE_META_LINE_PATTERN.match(stripped)
            or BLOCKQUOTE_LINE_PATTERN.match(stripped)
            or RAW_TABLE_DATA_LINE_PATTERN.match(stripped)
            or MARKDOWN_IMAGE_LINE_PATTERN.match(stripped)
        )

    def _extract_references_from_line(self, line: str) -> list[_Reference]:
        references: list[_Reference] = []

        for match in FIGURE_REF_PATTERN.finditer(line):
            number = int(match.group(1))
            raw_labels = match.group(2) or ""
            labels = tuple(label.lower() for label in re.findall(r"[a-z]", raw_labels))
            references.append(
                _Reference(kind="figure", number=number, sub_labels=labels)
            )

        for match in TABLE_REF_PATTERN.finditer(line):
            number = int(match.group(1))
            references.append(_Reference(kind="table", number=number))

        return references

    def _build_coverage(
        self,
        content_list_path: str | None,
        references: list[_Reference],
        attached_keys: set[str],
        missing_keys: set[str],
    ) -> dict[str, object]:
        detected_refs = [ref.display_label for ref in references]

        attached_refs: list[str] = []
        missing_refs: list[str] = []
        for ref in references:
            if ref.key in attached_keys:
                attached_refs.append(ref.display_label)
            elif ref.key in missing_keys:
                missing_refs.append(ref.display_label)

        detected_count = len(detected_refs)
        attached_count = len(attached_refs)
        coverage_ratio = (
            1.0 if detected_count == 0 else round(attached_count / detected_count, 3)
        )

        return {
            "content_list_path": content_list_path,
            "detected_refs": detected_refs,
            "attached_refs": attached_refs,
            "missing_refs": missing_refs,
            "detected_count": detected_count,
            "attached_count": attached_count,
            "missing_count": len(missing_refs),
            "coverage_ratio": coverage_ratio,
            "is_complete": len(missing_refs) == 0,
        }

    def _collect_existing_attachment_labels(self, summary_text: str) -> set[str]:
        labels: set[str] = set()
        for line in summary_text.splitlines():
            match = ATTACHMENT_HEADER_PATTERN.match(line)
            if match:
                labels.add(match.group(1).strip())
        return labels

    def _collect_existing_asset_paths(self, summary_text: str) -> set[str]:
        paths: set[str] = set()
        for line in summary_text.splitlines():
            for match in MARKDOWN_IMAGE_PATH_PATTERN.finditer(line):
                path = match.group(1).strip()
                if path:
                    paths.add(path)
        return paths

    def _collect_image_paths_from_lines(self, lines: list[str]) -> set[str]:
        paths: set[str] = set()
        for line in lines:
            for match in MARKDOWN_IMAGE_PATH_PATTERN.finditer(line):
                path = match.group(1).strip()
                if path:
                    paths.add(path)
        return paths

    def _pick_content_list_path(self, extract_dir: Path) -> Path | None:
        if not extract_dir.exists():
            return None

        candidate_patterns = [
            "*_content_list.json",
            "content_list.json",
            "content_list_v2.json",
        ]
        for pattern in candidate_patterns:
            matches = sorted(extract_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _collect_typed_items(self, payload: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                if isinstance(node_type, str):
                    items.append(node)
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for value in node:
                    _walk(value)

        _walk(payload)
        return items

    def _parse_image_candidate(
        self, item: dict[str, Any], item_index: int
    ) -> _ImageCandidate | None:
        img_path = self._extract_path(item, field_name="img_path")
        if not img_path:
            return None

        caption_lines = self._extract_text_lines(item, field_name="image_caption")
        footnote_lines = self._extract_text_lines(item, field_name="image_footnote")
        all_lines = [*caption_lines, *footnote_lines]
        full_caption_text = " ".join(all_lines)
        figure_number = self._extract_number(FIGURE_CAPTION_PATTERN, full_caption_text)

        sub_label = None
        for line in caption_lines:
            sub_match = SUBFIGURE_LABEL_PATTERN.match(line)
            if sub_match:
                sub_label = sub_match.group(1).lower()
                break

        return _ImageCandidate(
            item_index=item_index,
            page_idx=self._safe_int(item.get("page_idx"), default=-1),
            img_path=img_path,
            caption_lines=all_lines,
            figure_number=figure_number,
            sub_label=sub_label,
        )

    def _parse_table_asset(self, item: dict[str, Any]) -> _TableAsset | None:
        caption_lines = self._extract_text_lines(item, field_name="table_caption")
        footnote_lines = self._extract_text_lines(item, field_name="table_footnote")
        merged_caption = " ".join([*caption_lines, *footnote_lines]).strip()

        table_number = self._extract_number(TABLE_CAPTION_PATTERN, merged_caption)
        if table_number is None:
            return None

        table_body = item.get("table_body")
        if not isinstance(table_body, str):
            table_body = self._read_nested_str(item, "content", "html") or ""

        return _TableAsset(
            number=table_number,
            caption=merged_caption,
            image_path=self._extract_path(item, field_name="img_path"),
            table_body=table_body,
        )

    def _extract_path(self, item: dict[str, Any], field_name: str) -> str:
        direct_value = item.get(field_name)
        if isinstance(direct_value, str) and direct_value.strip():
            return direct_value.strip()

        nested_value = self._read_nested_str(item, "content", "image_source", "path")
        if nested_value:
            return nested_value

        return ""

    def _extract_text_lines(self, item: dict[str, Any], field_name: str) -> list[str]:
        direct_value = item.get(field_name)
        if direct_value is None:
            direct_value = self._read_nested(item, "content", field_name)

        if not isinstance(direct_value, list):
            return []

        lines: list[str] = []
        for entry in direct_value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    lines.append(text)
            elif isinstance(entry, dict):
                text = ""
                content = entry.get("content")
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(entry.get("text"), str):
                    text = str(entry["text"]).strip()
                if text:
                    lines.append(text)

        return lines

    def _infer_missing_figure_numbers(self, candidates: list[_ImageCandidate]) -> None:
        known_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.figure_number is not None
        ]
        if not known_indices:
            return

        for index, candidate in enumerate(candidates):
            if candidate.figure_number is not None:
                continue

            best_number: int | None = None
            best_distance: int | None = None

            for known_index in known_indices:
                known_candidate = candidates[known_index]
                if known_candidate.figure_number is None:
                    continue

                if candidate.page_idx >= 0 and known_candidate.page_idx >= 0:
                    if candidate.page_idx != known_candidate.page_idx:
                        continue

                distance = abs(index - known_index)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_number = known_candidate.figure_number

            if (
                best_number is not None
                and best_distance is not None
                and best_distance <= MAX_NEIGHBOR_DISTANCE
            ):
                candidate.figure_number = best_number

    def _resolve_relative_asset_path(
        self,
        extract_dir: Path,
        raw_path: str,
        summary_path: Path,
    ) -> str | None:
        if not raw_path:
            return None

        candidate_path = Path(raw_path)
        if not candidate_path.is_absolute():
            candidate_path = extract_dir / candidate_path

        if not candidate_path.exists():
            return None

        summary_asset_dir = summary_path.with_suffix("")
        destination_path = self._build_destination_asset_path(
            source_path=candidate_path,
            raw_path=raw_path,
            extract_dir=extract_dir,
            summary_asset_dir=summary_asset_dir,
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not destination_path.exists():
            shutil.copy2(candidate_path, destination_path)

        rel_path = os.path.relpath(destination_path, start=summary_path.parent)
        return rel_path.replace(os.sep, "/")

    def _build_destination_asset_path(
        self,
        source_path: Path,
        raw_path: str,
        extract_dir: Path,
        summary_asset_dir: Path,
    ) -> Path:
        raw_asset_path = Path(raw_path)
        if raw_asset_path.is_absolute():
            relative_asset_path = self._relative_asset_path_for_absolute_source(
                source_path=source_path,
                extract_dir=extract_dir,
            )
        else:
            relative_asset_path = raw_asset_path

        safe_parts = [
            part
            for part in relative_asset_path.parts
            if part and part not in {".", ".."}
        ]
        if not safe_parts:
            safe_parts = [source_path.name]

        return summary_asset_dir.joinpath(*safe_parts)

    def _relative_asset_path_for_absolute_source(
        self,
        source_path: Path,
        extract_dir: Path,
    ) -> Path:
        try:
            return source_path.relative_to(extract_dir)
        except ValueError:
            return Path("external") / source_path.name

    def _extract_number(self, pattern: re.Pattern[str], text: str) -> int | None:
        match = pattern.search(text)
        if not match:
            return None
        return int(match.group(1))

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _read_nested(self, data: dict[str, Any], *keys: str) -> Any:
        current: Any = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _read_nested_str(self, data: dict[str, Any], *keys: str) -> str:
        value = self._read_nested(data, *keys)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""
