"""Rich-based progress tracking for PaperWise."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ProcessingStage:
    """Represents a single processing stage."""

    name: str
    icon: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: float | None = None
    end_time: float | None = None
    details: str = ""

    @property
    def display_status(self) -> str:
        status_icons = {
            "pending": "⏸️",
            "running": "⏳",
            "completed": "✅",
            "failed": "❌",
        }
        return status_icons.get(self.status, "⏸️")

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def start(self) -> None:
        self.status = "running"
        self.start_time = time.time()

    def complete(self) -> None:
        self.status = "completed"
        self.end_time = time.time()

    def fail(self) -> None:
        self.status = "failed"
        self.end_time = time.time()


@dataclass
class FileProgress:
    """Tracks progress for a single file."""

    filename: str
    stages: list[ProcessingStage] = field(default_factory=list)
    token_usage: int = 0
    estimated_cost: float = 0.0
    overall_start_time: float = field(default_factory=time.time)

    def get_overall_progress(self) -> float:
        if not self.stages:
            return 0.0
        completed = sum(1 for s in self.stages if s.status == "completed")
        return completed / len(self.stages)


class RichProgressTracker:
    """Main progress tracker using Rich."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.files: list[FileProgress] = []
        self.current_file_idx: int = 0
        self._live: Live | None = None
        self._start_time: float = time.time()

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def add_file(self, filename: str, stages: list[str]) -> int:
        """Add a new file to track. Returns file index."""
        stage_objects = [
            ProcessingStage(
                name=name,
                icon="🔧" if "MinerU" in name else "📝",
            )
            for name in stages
        ]
        file_progress = FileProgress(
            filename=filename,
            stages=stage_objects,
        )
        self.files.append(file_progress)
        return len(self.files) - 1

    def update_stage(
        self,
        file_idx: int,
        stage_name: str,
        status: str,
        details: str = "",
    ) -> None:
        """Update stage status."""
        file_prog = self.files[file_idx]
        for stage in file_prog.stages:
            if stage.name == stage_name:
                stage.status = status
                stage.details = details
                if status == "running" and stage.start_time is None:
                    stage.start_time = time.time()
                elif status in ("completed", "failed") and stage.end_time is None:
                    stage.end_time = time.time()
                break
        if self._live:
            self._live.update(self._render())

    def update_token_usage(self, file_idx: int, tokens: int, cost: float) -> None:
        """Update token usage info."""
        self.files[file_idx].token_usage = tokens
        self.files[file_idx].estimated_cost = cost
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        """Render the full progress display."""
        main_content = self._build_main_content()
        total_elapsed = time.time() - self._start_time

        footer = Text()
        footer.append("💰 Token Usage: ", style="bold yellow")
        total_tokens = sum(f.token_usage for f in self.files)
        total_cost = sum(f.estimated_cost for f in self.files)
        footer.append(f"{total_tokens:,}", style="yellow")
        footer.append(" | Est. Cost: ", style="dim")
        footer.append(f"${total_cost:.4f}", style="green")
        footer.append(f" | Elapsed: {self._format_time(total_elapsed)}", style="dim")

        content = Table(show_header=False, box=None, padding=(0, 0))
        content.add_column("Main", ratio=1)
        content.add_row(main_content)
        content.add_row(footer)

        return Panel(
            content,
            title="[bold blue]PaperWise Processing[/bold blue]",
            border_style="blue",
        )

    def _build_main_content(self) -> Table:
        """Build the main progress table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Content", ratio=1)

        for idx, file_prog in enumerate(self.files):
            is_current = idx == self.current_file_idx
            prefix = "▶️" if is_current else "  "

            # File header
            file_text = Text(f"{prefix} 📄 {file_prog.filename}")
            if is_current:
                file_text.stylize("bold cyan")
            table.add_row(file_text)

            # Stages
            for stage in file_prog.stages:
                stage_text = self._format_stage(stage, is_current)
                table.add_row(stage_text)

        return table

    def _format_stage(self, stage: ProcessingStage, is_active: bool) -> Text:
        """Format a single stage line."""
        indent = "   ├── " if is_active else "      "
        status = stage.display_status

        text = Text(f"{indent}{status} {stage.name}")

        if stage.details:
            text.append(f"  ({stage.details})", style="dim")

        if stage.status == "running":
            elapsed = stage.elapsed
            text.append(f"  [{self._format_time(elapsed)}]", style="dim")
            text.stylize("yellow")
        elif stage.status == "completed":
            text.stylize("green")
        elif stage.status == "failed":
            text.stylize("red")

        return text

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time as human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"


class StageUpdater:
    """Helper to update stages for a specific file."""

    def __init__(self, tracker: RichProgressTracker, file_idx: int) -> None:
        self.tracker = tracker
        self.file_idx = file_idx

    def start(self, stage_name: str, details: str = "") -> None:
        """Mark stage as running."""
        self.tracker.update_stage(self.file_idx, stage_name, "running", details)

    def complete(self, stage_name: str, details: str = "") -> None:
        """Mark stage as completed."""
        self.tracker.update_stage(self.file_idx, stage_name, "completed", details)

    def fail(self, stage_name: str, details: str = "") -> None:
        """Mark stage as failed."""
        self.tracker.update_stage(self.file_idx, stage_name, "failed", details)

    def update_details(self, stage_name: str, details: str) -> None:
        """Update stage details while keeping status."""
        file_prog = self.tracker.files[self.file_idx]
        for stage in file_prog.stages:
            if stage.name == stage_name:
                stage.details = details
                break
        if self.tracker._live:
            self.tracker._live.update(self.tracker._render())


class track_processing:
    """Context manager for tracking file processing."""

    STAGES = [
        "MinerU Upload",
        "MinerU Parsing",
        "Story Planning",
        "Layer 1/5 (TL;DR)",
        "Layer 2/5 (Motivation)",
        "Layer 3/5 (Method)",
        "Layer 4/5 (Results)",
        "Layer 5/5 (Insights)",
        "Review",
        "Final Rewrite",
        "Save Results",
    ]

    def __init__(self, filename: str, tracker: RichProgressTracker):
        self.filename = filename
        self.tracker = tracker
        self.file_idx: int | None = None
        self.stage_updater: StageUpdater | None = None

    def __enter__(self) -> StageUpdater:
        self.file_idx = self.tracker.add_file(self.filename, self.STAGES)
        self.tracker.current_file_idx = self.file_idx
        self.stage_updater = StageUpdater(self.tracker, self.file_idx)
        return self.stage_updater

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# Compatibility: Provide a tqdm-like fallback when Rich is not available or disabled
class TqdmProgressAdapter:
    """Adapter that provides a similar interface but uses tqdm."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._tracker: RichProgressTracker | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def add_file(self, filename: str, stages: list[str]) -> int:
        return 0

    def update_stage(self, file_idx: int, stage_name: str, status: str, details: str = "") -> None:
        pass

    def update_token_usage(self, file_idx: int, tokens: int, cost: float) -> None:
        pass
