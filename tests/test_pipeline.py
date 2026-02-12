import json
from pathlib import Path

from papersummarizer.models import (
    LLMCallUsage,
    ParsedPaper,
    SummarizationResult,
    SummarizationTokenUsage,
)
from papersummarizer.pipeline import PaperSummarizationPipeline


def _complete_five_layer_summary() -> str:
    return """# 预算可控推理：在数学任务中提升准确率并降低超预算率
> 标题：Budget-Aware Reasoning via Persistent Control Signals

## 1) TL;DR
One-Liner：这篇论文把预算控制从提示词升级到训练目标，在预算受限场景依然保持较高准确率。

---

## 2) Motivation & Gap
痛点是旧方法在长链推理里预算信号会衰减，导致输出超长或提前截断。研究空白在于此前方案很少把预算控制作为可学习目标。核心洞察是把预算控制信号持续注入解码过程，并在训练中显式优化预算约束。

---

## 3) Method & Mechanism
方法架构见 Figure 1。数据流是输入问题与预算后进入控制模块，再进入主推理模块，最后输出答案与预算跟随率。创新点包括预算持续控制、分阶段训练和约束一致性校准模块。

---

## 4) Proof & Results
在 MATH-500 数据集上，相对 Baseline 提升 5.2%，达到接近 SOTA 的结果；相关对比见 Figure 2 和 Table 1。消融实验表明去掉预算持续注入后性能下降最明显。可视化案例显示在相同预算下新方法更少出现冗长推理。

---

## 5) Insights & Decision
局限在于跨领域迁移证据仍不足，且对推理框架改造有要求。未来方向是扩大到非数学任务并优化训练成本。建议投入：在预算受限且具备模型训练能力的团队中先做小规模验证，再决定是否全量上线。
""".strip()


class FakeParser:
    def parse_pdf(self, pdf_path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return ParsedPaper(
            pdf_path=pdf_path,
            markdown_text="# Title\n\nparsed text",
            artifact_dir=artifact_dir,
        )

    def parse_pdf_url(
        self, pdf_url: str, artifact_dir: Path, file_name: str | None = None
    ):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return ParsedPaper(
            pdf_path=Path(file_name or "remote.pdf"),
            markdown_text="# Title\n\nparsed text from url",
            artifact_dir=artifact_dir,
        )


class FakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        return f"summary for {paper_title}"


class FiveLayerFakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        return _complete_five_layer_summary()


class MissingLayerFakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        summary = _complete_five_layer_summary()
        return summary.replace("## 5) Insights & Decision", "## 5) Missing", 1)


class NoDecisionWithAnchorsFakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        return """# 预算可控推理：在数学任务中提升准确率并降低超预算率
> 标题：Budget-Aware Reasoning via Persistent Control Signals

## 1) TL;DR
One-Liner：这篇论文把预算控制从提示词升级到训练目标，在预算受限场景依然保持较高准确率。

---

## 2) Motivation & Gap
痛点是旧方法在长链推理里预算信号会衰减，导致输出超长或提前截断。研究空白在于此前方案很少把预算控制作为可学习目标。核心洞察是把预算控制信号持续注入解码过程，并在训练中显式优化预算约束。

---

## 3) Method & Mechanism
方法架构见 Figure 1。数据流是输入问题与预算后进入控制模块，再进入主推理模块，最后输出答案与预算跟随率。创新点包括预算持续控制、分阶段训练和约束一致性校准模块。

---

## 4) Proof & Results
在 MATH-500 数据集上，相对 Baseline 提升 5.2%，达到接近 SOTA 的结果；相关对比见 Figure 2 和 Table 1。消融实验表明去掉预算持续注入后性能下降最明显。可视化案例显示在相同预算下新方法更少出现冗长推理。
**证据锚点**：Figure 2、Table 1

---

## 5) Insights & Decision
局限在于跨领域迁移证据仍不足，且对推理框架改造有要求。未来方向是扩大到非数学任务并优化训练成本。
""".strip()


class MetricsFakeSummarizer:
    def summarize_with_metrics(self, paper_title: str, paper_text: str):
        usage = SummarizationTokenUsage(
            enabled=True,
            usage_available=True,
            steps=[
                LLMCallUsage(
                    step_name="story_planner",
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    usage_available=True,
                ),
                LLMCallUsage(
                    step_name="layer_1_hook_tldr",
                    prompt_tokens=80,
                    completion_tokens=30,
                    total_tokens=110,
                    usage_available=True,
                ),
            ],
        )
        return SummarizationResult(
            summary_text=_complete_five_layer_summary(),
            token_usage=usage,
        )


class EvidenceFakeParser:
    def parse_pdf(self, pdf_path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = artifact_dir / "mineru_extract"
        images_dir = extract_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        (images_dir / "fig2.jpg").write_bytes(b"jpeg")
        (images_dir / "table1.jpg").write_bytes(b"jpeg")

        content_list = [
            {
                "type": "image",
                "img_path": "images/fig2.jpg",
                "image_caption": ["Figure 2 | Accuracy across budgets."],
                "image_footnote": [],
                "page_idx": 6,
            },
            {
                "type": "table",
                "img_path": "images/table1.jpg",
                "table_caption": ["Table 1 | AIME results."],
                "table_footnote": [],
                "table_body": "<table><tr><td>16.25</td></tr></table>",
                "page_idx": 7,
            },
        ]
        (extract_dir / "sample_content_list.json").write_text(
            json.dumps(content_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ParsedPaper(
            pdf_path=pdf_path,
            markdown_text="# Title\n\nparsed text",
            artifact_dir=artifact_dir,
        )


class FiveLayerFakeSummarizerWithEvidenceRefs:
    def summarize(self, paper_title: str, paper_text: str):
        return """# 预算控制模型：结果更稳
> 标题：Paper

## 1) TL;DR
One-Liner：如 Figure 2 所示，预算控制更稳定。

---

## 2) Motivation & Gap
痛点明显，研究空白清晰，核心洞察可行。

---

## 3) Method & Mechanism
架构在 Figure 2 中给出。

---

## 4) Proof & Results
在数据集上相对 baseline 提升 3%，见 Table 1。

---

## 5) Insights & Decision
局限、未来方向与建议投入。
""".strip()


class MetadataRichFakeParser:
    def parse_pdf(self, pdf_path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        markdown_text = """# CoMAS: Collaborative Multi-Agent Self-Evolution

Alice Zhang, Bob Li, Carol Wang
Tsinghua University, Shanghai AI Lab
arXiv:2510.08529 [cs.AI], preprint

## Abstract

paper body
"""
        return ParsedPaper(
            pdf_path=pdf_path,
            markdown_text=markdown_text,
            artifact_dir=artifact_dir,
        )


class SpySummarizer:
    def __init__(self) -> None:
        self.received_titles: list[str] = []

    def summarize(self, paper_title: str, paper_text: str):
        self.received_titles.append(paper_title)
        return _complete_five_layer_summary()


class HeaderFixFakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        return """# CoMAS：通过智能体间交互奖励实现去中心化自进化的多智能体系统
> 标题：2510.08529

## 1) TL;DR
这篇论文提出了一个通过多智能体交互奖励驱动的训练框架。

---

## 2) Motivation & Gap
现有基于强化学习的智能体进化方法在奖励信号来源上存在根本性局限。如图1所示，当前主流范式主要依赖两类信号：一是外部奖励，通常源自基于规则的验证器或专用奖励模型，这类方法虽能提供明确监督，但无法适用于难以验证答案的开放域问题；二是单智能体内在奖励，通过自信度、语义熵或多数投票等内部信号驱动学习，虽摆脱了外部监督，却局限于个体模型的自我评判，缺乏群体协作带来的多样性增益。

---

## 3) Method & Mechanism
方法见 Figure 1。

---

## 4) Proof & Results
结果见 Figure 2 和 Table 1。

---

## 5) Insights & Decision
局限和未来方向仍需验证。
""".strip()


class NoPublishInfoFakeParser:
    def parse_pdf(self, pdf_path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        markdown_text = """# CoMAS: Collaborative Multi-Agent Self-Evolution

Alice Zhang, Bob Li, Carol Wang
Tsinghua University, Shanghai AI Lab

## Abstract

paper body
"""
        return ParsedPaper(
            pdf_path=pdf_path,
            markdown_text=markdown_text,
            artifact_dir=artifact_dir,
        )


class NoisyAuthorFakeParser:
    def parse_pdf(self, pdf_path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        markdown_text = """# COMAS: CO-EVOLVING MULTI-AGENT SYSTEMS VIA INTERACTION REWARDS

Xiangyuan Xue $^{1,2}$ Yifan Zhou $^{3}$ Guibin Zhang $^{4}$ Zaibin Zhang $^{5,6}$ Yijiang Li $^{7}$ Chen Zhang $^{2,8}$ Zhenfei Yin $^{6\text{圆}}$ Philip Torr $^{6}$ Wanli Ouyang $^{1,2}$ Lei Bai $^{2\text{圆}}$

<sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>Shanghai Artificial Intelligence Laboratory

## Abstract

paper body
"""
        return ParsedPaper(
            pdf_path=pdf_path,
            markdown_text=markdown_text,
            artifact_dir=artifact_dir,
        )


class TldrFormattingFakeSummarizer:
    def summarize(self, paper_title: str, paper_text: str):
        return """# 示例标题：强调交互奖励稳定提升
> 标题：COMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards

## 1) TL;DR
一句话总结：CoMAS通过多智能体相互评分生成内在奖励，实现无需外部验证器的自进化。

一页卡片：主流强化学习依赖外部验证器或单模型自评，CoMAS通过“方案提出→批判评估→LLM裁判打分”闭环，让异构智能体在不共享参数下协同优化。

---

## 2) Motivation & Gap
痛点和研究空白清晰。

---

## 3) Method & Mechanism
方法见 Figure 1。

---

## 4) Proof & Results
结果见 Table 1。

---

## 5) Insights & Decision
局限与未来方向。
""".strip()


def test_pipeline_creates_outputs(tmp_path):
    pdf_path = tmp_path / "papers" / "a.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=FakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    results = pipeline.run(
        input_path=pdf_path.parent,
        max_files=None,
        skip_existing=False,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert (tmp_path / "outputs" / "parsed_markdown" / "a.md").exists()
    assert (tmp_path / "outputs" / "summaries" / "a.md").exists()
    assert (tmp_path / "outputs" / "metadata" / "a.json").exists()

    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "a.json").read_text(encoding="utf-8")
    )
    assert metadata["summary_format_version"] == "five_layers_v1"
    assert "summary_token_usage" in metadata
    assert metadata["summary_coverage"]["is_complete"] is False


def test_pipeline_creates_outputs_for_urls(tmp_path):
    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=FakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    results = pipeline.run_urls(
        pdf_urls=["https://arxiv.org/pdf/2404.13501.pdf"],
        skip_existing=False,
    )

    assert len(results) == 1
    assert results[0].success is True
    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "2404.13501.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["summary_coverage"]["is_complete"] is False


def test_pipeline_marks_complete_summary_coverage_and_style(tmp_path):
    pdf_path = tmp_path / "papers" / "b.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=FiveLayerFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "b.json").read_text(encoding="utf-8")
    )
    assert metadata["summary_coverage"]["is_complete"] is True
    assert metadata["summary_style_coverage"]["is_complete"] is True


def test_pipeline_marks_incomplete_when_layer_missing(tmp_path):
    pdf_path = tmp_path / "papers" / "c.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=MissingLayerFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "c.json").read_text(encoding="utf-8")
    )
    assert metadata["summary_coverage"]["is_complete"] is False
    assert "insights_decision" in metadata["summary_coverage"]["missing_layers"]


def test_pipeline_strips_evidence_anchor_lines_and_no_decision_requirement(tmp_path):
    pdf_path = tmp_path / "papers" / "f.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=NoDecisionWithAnchorsFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_text = (tmp_path / "outputs" / "summaries" / "f.md").read_text(
        encoding="utf-8"
    )
    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "f.json").read_text(encoding="utf-8")
    )

    assert "证据锚点" not in summary_text
    assert metadata["summary_style_coverage"]["is_complete"] is True


def test_pipeline_writes_token_usage_from_summarizer_metrics(tmp_path):
    pdf_path = tmp_path / "papers" / "d.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=MetricsFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "d.json").read_text(encoding="utf-8")
    )
    usage = metadata["summary_token_usage"]
    assert usage["enabled"] is True
    assert usage["usage_available"] is True
    assert usage["aggregate"]["total_tokens"] == 230
    assert usage["aggregate"]["step_count"] == 2


def test_pipeline_enriches_summary_with_evidence_assets(tmp_path):
    pdf_path = tmp_path / "papers" / "e.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=EvidenceFakeParser(),
        summarizer=FiveLayerFakeSummarizerWithEvidenceRefs(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_text = (tmp_path / "outputs" / "summaries" / "e.md").read_text(
        encoding="utf-8"
    )
    metadata = json.loads(
        (tmp_path / "outputs" / "metadata" / "e.json").read_text(encoding="utf-8")
    )

    assert "![Figure 2](e/images/fig2.jpg)" in summary_text
    assert "![Table 1](e/images/table1.jpg)" in summary_text
    assert (tmp_path / "outputs" / "summaries" / "e" / "images" / "fig2.jpg").exists()
    assert (tmp_path / "outputs" / "summaries" / "e" / "images" / "table1.jpg").exists()
    assert metadata["summary_evidence_coverage"]["detected_count"] == 2
    assert metadata["summary_evidence_coverage"]["attached_count"] == 2
    assert metadata["summary_evidence_coverage"]["is_complete"] is True


def test_pipeline_uses_parsed_markdown_title_for_summarizer(tmp_path):
    pdf_path = tmp_path / "papers" / "2510.08529.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    spy_summarizer = SpySummarizer()
    pipeline = PaperSummarizationPipeline(
        parser=MetadataRichFakeParser(),
        summarizer=spy_summarizer,
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    assert spy_summarizer.received_titles == [
        "CoMAS: Collaborative Multi-Agent Self-Evolution"
    ]


def test_pipeline_injects_paper_metadata_header_and_readability_breaks(tmp_path):
    pdf_path = tmp_path / "papers" / "2510.08529.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=MetadataRichFakeParser(),
        summarizer=HeaderFixFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_path = tmp_path / "outputs" / "summaries" / "2510.08529.md"
    metadata_path = tmp_path / "outputs" / "metadata" / "2510.08529.json"
    summary_text = summary_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "> 标题：CoMAS: Collaborative Multi-Agent Self-Evolution" in summary_text
    assert "> 作者：Alice Zhang, Bob Li, Carol Wang" in summary_text
    assert "> 单位：Tsinghua University, Shanghai AI Lab" in summary_text
    assert "> 发布信息：arXiv（preprint）" in summary_text
    assert (
        "**现有基于强化学习的智能体进化方法在奖励信号来源上存在根本性局限。**"
        in summary_text
    )
    assert "如图1所示，当前主流范式主要依赖两类信号" in summary_text

    assert (
        metadata["paper_metadata"]["title"]
        == "CoMAS: Collaborative Multi-Agent Self-Evolution"
    )
    assert metadata["paper_metadata"]["authors"] == [
        "Alice Zhang",
        "Bob Li",
        "Carol Wang",
    ]
    assert metadata["paper_metadata"]["affiliations"] == [
        "Tsinghua University, Shanghai AI Lab"
    ]
    assert metadata["paper_metadata"]["publication_platform"] == "arXiv"
    assert metadata["paper_metadata"]["publication_status"] == "preprint"


def test_pipeline_omits_publish_info_when_platform_missing(tmp_path):
    pdf_path = tmp_path / "papers" / "no-publish-info.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=NoPublishInfoFakeParser(),
        summarizer=HeaderFixFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_path = tmp_path / "outputs" / "summaries" / "no-publish-info.md"
    summary_text = summary_path.read_text(encoding="utf-8")

    assert "> 作者：Alice Zhang, Bob Li, Carol Wang" in summary_text
    assert "> 单位：Tsinghua University, Shanghai AI Lab" in summary_text
    assert "> 发布信息：" not in summary_text


def test_pipeline_cleans_noisy_superscript_authors(tmp_path):
    pdf_path = tmp_path / "papers" / "noisy-authors.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=NoisyAuthorFakeParser(),
        summarizer=FiveLayerFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    metadata_path = tmp_path / "outputs" / "metadata" / "noisy-authors.json"
    summary_path = tmp_path / "outputs" / "summaries" / "noisy-authors.md"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary_text = summary_path.read_text(encoding="utf-8")

    assert metadata["paper_metadata"]["authors"] == [
        "Xiangyuan Xue",
        "Yifan Zhou",
        "Guibin Zhang",
        "Zaibin Zhang",
        "Yijiang Li",
        "Chen Zhang",
        "Zhenfei Yin",
        "Philip Torr",
        "Wanli Ouyang",
        "Lei Bai",
    ]
    assert "> 作者：Xiangyuan Xue $^{1,2}$" in summary_text
    assert "Zhenfei Yin $^{6}$" in summary_text
    assert "Lei Bai $^{2}$" in summary_text
    assert "\\text{" not in summary_text
    assert "圆" not in summary_text


def test_pipeline_formats_uppercase_original_title_to_title_case(tmp_path):
    pdf_path = tmp_path / "papers" / "title-case.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=NoisyAuthorFakeParser(),
        summarizer=FiveLayerFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_path = tmp_path / "outputs" / "summaries" / "title-case.md"
    summary_text = summary_path.read_text(encoding="utf-8")

    assert (
        "> 标题：COMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards"
        in summary_text
    )


def test_pipeline_formats_tldr_one_liner_and_expansion_block(tmp_path):
    pdf_path = tmp_path / "papers" / "tldr-format.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pipeline = PaperSummarizationPipeline(
        parser=FakeParser(),
        summarizer=TldrFormattingFakeSummarizer(),
        output_dir=tmp_path / "outputs",
    )

    pipeline.run(input_path=pdf_path.parent, max_files=None, skip_existing=False)

    summary_path = tmp_path / "outputs" / "summaries" / "tldr-format.md"
    summary_text = summary_path.read_text(encoding="utf-8")

    assert "一句话总结：\n\n> [!TIP]" in summary_text
    assert "一页卡片：" not in summary_text
    assert "展开来讲：" in summary_text
