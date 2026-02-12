# PaperWise Rich 进度展示改进计划

## 🎯 当前问题分析

### 1. 进度展示现状
- 仅使用 `tqdm` 显示文件级进度：`Processing PDFs: 0%| | 0/1 [00:00<?, ?it/s]`
- **MinerU 解析阶段**：轮询等待没有任何反馈，看起来像"卡死"
- **LLM 总结阶段**：多阶段流水线（story planner → 5 layers → review → rewrite）完全黑盒

### 2. 卡点位置
| 阶段 | 耗时估算 | 当前状态 |
|------|----------|----------|
| MinerU 上传 | 1-5s | 无反馈 |
| MinerU 解析轮询 | 10-60s | **完全无进度，像卡死** |
| Story Planner | 10-30s | 无反馈 |
| 5层逐层生成 | 5 × 10-20s = 50-100s | **最大黑盒，不知道在哪一层** |
| Review | 10-20s | 无反馈 |
| Rewrite | 15-30s | 无反馈 |
| **总计** | **2-4分钟** | **用户只看到0%或100%** |

## 🎨 Rich 改进方案

### 1. 整体架构设计

```
┌─────────────────────────────────────────────────────────────┐
│  PaperWise Processing                                        │
│  ═══════════════════════════════════════════════════════    │
│                                                              │
│  📄 test_paper.pdf                                          │
│  ├── 🔧 MinerU Parsing      [████████░░] 80%  ETA: 12s     │
│  │   ├── Uploading...        ✅ 2.1s                       │
│  │   └── Waiting for result  ⏳ 18s (polling...)           │
│  ├── 📝 LLM Summarizing     [████░░░░░░] 40%  ETA: 45s     │
│  │   ├── Story Planning      ✅ 8.5s  (1,234 tokens)       │
│  │   ├── Layer 1/5 (TL;DR)   ✅ 6.2s                       │
│  │   ├── Layer 2/5 (Motivation) ⏳ Generating...           │
│  │   ├── Layer 3/5 (Method)   ⏸️  Pending...               │
│  │   ├── Layer 4/5 (Results)   ⏸️  Pending...              │
│  │   ├── Layer 5/5 (Insights)  ⏸️  Pending...              │
│  │   ├── Review               ⏸️  Pending...               │
│  │   └── Final Rewrite        ⏸️  Pending...               │
│  └── 💾 Saving Results       ⏸️  Pending...                │
│                                                              │
│  📊 Token Usage: 4,567 tokens | 💰 Est. Cost: $0.023      │
│  🕐 Elapsed: 45s | ⏱️ ETA: 90s                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 技术实现方案

#### 2.1 新增文件：`progress.py`

```python
"""Rich-based progress tracking for PaperWise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class ProcessingStage:
    """Represents a single processing stage."""
    name: str
    icon: str
    status: str = "pending"  # pending, running, completed, failed
    elapsed: float = 0.0
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


@dataclass
class FileProgress:
    """Tracks progress for a single file."""
    filename: str
    stages: list[ProcessingStage] = field(default_factory=list)
    token_usage: int = 0
    estimated_cost: float = 0.0
    
    def get_overall_progress(self) -> float:
        if not self.stages:
            return 0.0
        completed = sum(1 for s in self.stages if s.status == "completed")
        return completed / len(self.stages)


class RichProgressTracker:
    """Main progress tracker using Rich."""
    
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.files: list[FileProgress] = []
        self.current_file_idx: int = 0
        self._live: Live | None = None
        
    def start(self):
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()
        
    def stop(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            
    def add_file(self, filename: str, stages: list[str]) -> int:
        """Add a new file to track. Returns file index."""
        stage_objects = [
            ProcessingStage(name=name, icon="🔧" if "MinerU" in name else "📝")
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
    ):
        """Update stage status."""
        file_prog = self.files[file_idx]
        for stage in file_prog.stages:
            if stage.name == stage_name:
                stage.status = status
                stage.details = details
                break
        if self._live:
            self._live.update(self._render())
            
    def update_token_usage(self, file_idx: int, tokens: int, cost: float):
        """Update token usage info."""
        self.files[file_idx].token_usage = tokens
        self.files[file_idx].estimated_cost = cost
        if self._live:
            self._live.update(self._render())
            
    def _render(self) -> Layout:
        """Render the full progress display."""
        layout = Layout()
        
        # Main progress panel
        main_content = self._build_main_content()
        layout.update(Panel(
            main_content,
            title="PaperWise Processing",
            border_style="blue",
        ))
        
        return layout
        
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
                
            # Token usage summary
            if file_prog.token_usage > 0:
                usage_text = Text(
                    f"   💰 Tokens: {file_prog.token_usage:,} "
                    f"(~${file_prog.estimated_cost:.4f})",
                    style="dim"
                )
                table.add_row(usage_text)
                
        return table
        
    def _format_stage(self, stage: ProcessingStage, is_active: bool) -> Text:
        """Format a single stage line."""
        indent = "   ├── " if is_active else "      "
        status = stage.display_status
        
        text = Text(f"{indent}{status} {stage.name}")
        
        if stage.details:
            text.append(f"  ({stage.details})", style="dim")
            
        if stage.status == "running":
            text.stylize("yellow")
        elif stage.status == "completed":
            text.stylize("green")
        elif stage.status == "failed":
            text.stylize("red")
            
        return text


# Context manager for easy usage
class track_processing:
    """Context manager for tracking file processing."""
    
    def __init__(self, filename: str, tracker: RichProgressTracker):
        self.filename = filename
        self.tracker = tracker
        self.file_idx: int | None = None
        
    def __enter__(self):
        stages = [
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
        self.file_idx = self.tracker.add_file(self.filename, stages)
        self.tracker.current_file_idx = self.file_idx
        return StageUpdater(self.tracker, self.file_idx)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class StageUpdater:
    """Helper to update stages for a specific file."""
    
    def __init__(self, tracker: RichProgressTracker, file_idx: int):
        self.tracker = tracker
        self.file_idx = file_idx
        
    def start(self, stage_name: str, details: str = ""):
        """Mark stage as running."""
        self.tracker.update_stage(self.file_idx, stage_name, "running", details)
        
    def complete(self, stage_name: str, details: str = ""):
        """Mark stage as completed."""
        self.tracker.update_stage(self.file_idx, stage_name, "completed", details)
        
    def fail(self, stage_name: str, details: str = ""):
        """Mark stage as failed."""
        self.tracker.update_stage(self.file_idx, stage_name, "failed", details)
```

#### 2.2 修改 `pipeline.py` 集成进度跟踪

```python
# 在 PaperSummarizationPipeline 中添加进度跟踪支持

from .progress import RichProgressTracker, track_processing

class PaperSummarizationPipeline:
    def __init__(
        self,
        parser: MinerUPdfParser,
        summarizer: OpenAISummarizer,
        output_dir: Path,
        evidence_enricher: SummaryEvidenceEnricher | None = None,
        summary_format: str = "five_layers_v1",
        progress_tracker: RichProgressTracker | None = None,  # 新增
    ) -> None:
        # ... 现有代码 ...
        self.progress_tracker = progress_tracker
        
    def process_one(
        self,
        pdf_path: Path,
        input_root: Path,
        skip_existing: bool,
    ) -> PipelineResult:
        # 使用进度跟踪上下文
        if self.progress_tracker:
            with track_processing(pdf_path.name, self.progress_tracker) as stage:
                return self._process_with_progress(
                    pdf_path, input_root, skip_existing, stage
                )
        else:
            # 回退到无进度模式
            return self._process_legacy(pdf_path, input_root, skip_existing)
            
    def _process_with_progress(
        self,
        pdf_path: Path,
        input_root: Path,
        skip_existing: bool,
        stage: StageUpdater,
    ) -> PipelineResult:
        # 在每个关键步骤调用 stage.start() 和 stage.complete()
        # 例如：
        stage.start("MinerU Upload", f"{pdf_path.stat().st_size / 1024:.1f} KB")
        # ... 上传代码 ...
        stage.complete("MinerU Upload", "Uploaded successfully")
        
        stage.start("MinerU Parsing", "Waiting for MinerU...")
        # ... 轮询代码 ...
        stage.complete("MinerU Parsing", f"{len(parsed_text):,} chars extracted")
```

#### 2.3 修改 `mineru_client.py` 添加轮询进度回调

```python
# 添加可选的进度回调参数

def wait_for_batch(
    self,
    batch_id: str,
    poll_interval_sec: float = 2.0,
    timeout_sec: int = 900,
    progress_callback: Callable[[int, str], None] | None = None,  # 新增
) -> list[ExtractResultItem]:
    """Wait for batch completion with optional progress callback."""
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout_sec:
        attempt += 1
        results = self.get_batch_results(batch_id)
        
        # 计算进度
        total = len(results)
        done = sum(1 for r in results if r.state == "done")
        failed = sum(1 for r in results if r.state == "failed")
        
        if progress_callback:
            elapsed = int(time.time() - start_time)
            progress_callback(
                int(done / total * 100) if total > 0 else 0,
                f"Attempt {attempt} | Done: {done}/{total} | Elapsed: {elapsed}s"
            )
        
        if all(r.state in ("done", "failed") for r in results):
            return results
            
        time.sleep(poll_interval_sec)
    
    raise TimeoutError(f"Polling timed out after {timeout_sec}s")
```

#### 2.4 修改 `openai_summarizer.py` 添加分层进度

```python
# 在每层生成时报告进度

def _build_draft_summary(
    self,
    paper_title: str,
    paper_text: str,
    narrative_plan: str,
    progress_callback: Callable[[int, str], None] | None = None,  # 新增
) -> tuple[str, list[LLMCallUsage]]:
    """Build draft summary layer by layer with progress tracking."""
    layer_specs = get_five_layer_specs()
    usages: list[LLMCallUsage] = []
    layers: list[str] = []
    
    for idx, spec in enumerate(layer_specs, 1):
        if progress_callback:
            progress_callback(idx, f"Generating {spec.name}...")
        
        # ... 生成代码 ...
        
        if progress_callback:
            progress_callback(idx, f"✓ {spec.name} ({usage.total_tokens} tokens)")
    
    return "\n\n---\n\n".join(layers), usages
```

### 3. 最终 CLI 输出效果

```
$ paperwise test_paper.pdf

╭─────────────────────────────────────────────────────────────╮
│  PaperWise v0.1.0 - Processing 1 file                        │
╰─────────────────────────────────────────────────────────────╯

▶️ 📄 test_paper.pdf
   ├── ✅ MinerU Upload          (735 KB, 2.3s)
   ├── ⏳ MinerU Parsing         Attempt 15 | Done: 0/1 | Elapsed: 30s
   ├── ⏸️ Story Planning         Pending...
   ├── ⏸️ Layer 1/5 (TL;DR)      Pending...
   ├── ⏸️ Layer 2/5 (Motivation) Pending...
   ├── ⏸️ Layer 3/5 (Method)     Pending...
   ├── ⏸️ Layer 4/5 (Results)    Pending...
   ├── ⏸️ Layer 5/5 (Insights)   Pending...
   ├── ⏸️ Review                 Pending...
   ├── ⏸️ Final Rewrite         Pending...
   └── ⏸️ Save Results          Pending...

💰 Token Usage: 0 | Est. Cost: $0.0000 | Elapsed: 32s

═══════════════════════════════════════════════════════════════
✅ Finished! 1/1 files processed successfully
📁 Output: ./outputs/summaries/test_paper.md
💰 Total Tokens: 12,345 | Total Cost: ~$0.062
⏱️  Total Time: 2m 34s
```

### 4. 错误状态可视化

```
▶️ 📄 broken_paper.pdf
   ├── ✅ MinerU Upload          (128 KB, 1.2s)
   ├── ❌ MinerU Parsing         Failed: PDF is corrupted or encrypted
   ├── ⏸️ Story Planning         Skipped
   ...

═══════════════════════════════════════════════════════════════
⚠️  Finished with errors: 0/1 files processed successfully
❌ Failed: broken_paper.pdf - PDF is corrupted or encrypted
```

### 5. 批量处理模式

```
$ paperwise papers/ --max-files 5

╭─────────────────────────────────────────────────────────────╮
│  PaperWise v0.1.0 - Processing 5 files                       │
╰─────────────────────────────────────────────────────────────╯

▶️ 📄 paper_1.pdf (3/5)
   ├── ✅ MinerU Upload
   ├── ⏳ MinerU Parsing         Attempt 8 | Elapsed: 16s
   ...

⏸️ 📄 paper_2.pdf (Pending)
⏸️ 📄 paper_3.pdf (Pending)
⏸️ 📄 paper_4.pdf (Pending)
⏸️ 📄 paper_5.pdf (Pending)

Overall Progress: [████░░░░░░░░░░░░░░░░] 20% (1/5 files)

💰 Token Usage: 4,567 | Est. Cost: $0.023 | Elapsed: 45s
```

## 📋 实施步骤

### Phase 1: 基础框架
1. 添加 `rich` 到依赖
2. 创建 `progress.py` 基础类
3. 修改 `cli.py` 初始化进度跟踪器

### Phase 2: MinerU 进度
1. 修改 `mineru_client.py` 添加轮询回调
2. 修改 `mineru_parser.py` 报告进度

### Phase 3: LLM 进度
1. 修改 `openai_summarizer.py` 报告分层进度
2. 修改 `pipeline.py` 集成完整进度流

### Phase 4: 错误处理
1. 添加失败状态可视化
2. 添加重试进度显示
3. 网络超时可视化

## 🔧 依赖修改

```toml
# pyproject.toml
dependencies = [
    "openai>=1.30.0",
    "python-dotenv>=1.0.1",
    "requests>=2.32.0",
    "rich>=13.0.0",  # 新增
]
```

## 💡 额外建议

1. **添加 `--verbose` 模式**: 显示原始 API 响应
2. **添加 `--quiet` 模式**: 仅显示错误和最终结果
3. **添加 `--json` 输出**: 用于脚本集成
4. **网络诊断**: 当 MinerU 轮询超时时，显示诊断信息（如"检查网络连接"）

这个方案可以让用户一眼看出：
- ✅ 当前在哪一步
- ⏱️ 每一步花了多少时间
- 💰 累计 Token 消耗和成本
- ❌ 如果出错，具体在哪一步失败
