"""Prompt templates and rendering helpers."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a rigorous PhD-level research analyst writing Chinese technical briefs "
    "for engineering decision makers. Keep claims faithful to source evidence, "
    "make the narrative easy to follow, and emphasize actionable insight."
)

DEFAULT_STORY_PLANNER_SYSTEM_PROMPT = (
    "You are a narrative planning agent for research paper explainers. "
    "Produce a five-layer blueprint with evidence anchors, not final prose."
)

FIVE_LAYER_SPECS: list[dict[str, str]] = [
    {
        "key": "hook_tldr",
        "index": "1",
        "title": "TL;DR",
        "focus": "用最短语言讲清这篇论文值不值得深读",
        "requirements": (
            "先在最前面输出标题行 `# <核心贡献 + 应用场景/效果>` 与"
            " `> 标题：{{paper_title}}`、`> 作者：...`、`> 单位：...`、`> 发布信息：...`；"
            "再给出“一句话总结”和“展开来讲（易懂短版）”；"
            "明确这篇论文解决了什么痛点、用了什么方法、结果是否可靠。"
        ),
    },
    {
        "key": "motivation_gap",
        "index": "2",
        "title": "Motivation & Gap",
        "focus": "解释为什么需要这项研究",
        "requirements": (
            "交代旧方法痛点（效率/精度/数据依赖等）、研究空白、核心洞察；"
            "说明此前尝试为何不足。"
        ),
    },
    {
        "key": "method_mechanism",
        "index": "3",
        "title": "Method & Mechanism",
        "focus": "解释方法如何工作",
        "requirements": (
            "引用最关键架构图（Figure/Table/图/表）；按数据流解释输入到输出；"
            "列出1-3个关键创新并解释功能而不是只写公式。"
        ),
    },
    {
        "key": "proof_results",
        "index": "4",
        "title": "Proof & Results",
        "focus": "用证据证明方法有效",
        "requirements": (
            "给出SOTA对比（在X数据集上相对Y提升Z%）；"
            "说明消融实验最关键结论；给出可视化/案例分析要点。"
        ),
    },
    {
        "key": "insights_decision",
        "index": "5",
        "title": "Insights & Decision",
        "focus": "给出深度思考与可迁移启示",
        "requirements": (
            "覆盖局限性、未来方向、可迁移启示；"
            "不要输出“建议投入/暂缓投入/决策建议”这类建议性结论。"
        ),
    },
]

DEFAULT_STORY_PLANNER_TEMPLATE = """你将先输出“洋葱式五层总结蓝图”，不是最终总结正文。

必须满足：
- 严禁编造；没有证据就写 `[未在原文找到直接证据]`
- 每层给出至少 2 个证据锚点（Figure/Table/章节）
- 保持层间连贯，给出每层承接句
- 逻辑去重：若不同层出现同一结论，请合并到最合适层，不要重复铺陈
- 解释预埋：预判晦涩概念，在对应层预留“机制解释/直观类比”槽位，避免正文临时补救

输出结构（标题必须完全一致）：
[LAYER-1-TLDR]
- 标题候选（核心贡献 + 场景/效果）：
- 一句话总结候选：
- 展开来讲（易懂短版）：
- 证据锚点：

[LAYER-2-MOTIVATION-GAP]
- 痛点：
- 研究空白：
- 核心洞察：
- 证据锚点：

[LAYER-3-METHOD]
- 架构主线：
- 创新点1/2/3：
- 证据锚点：

[LAYER-4-PROOF]
- SOTA关键数字：
- 消融关键结论：
- 案例/可视化证据：
- 证据锚点：

[LAYER-5-INSIGHTS-DECISION]
- 局限与风险：
- 未来方向：
- 决策建议：
- 证据锚点：

[LAYER-BRIDGES]
- 1->2:
- 2->3:
- 3->4:
- 4->5:

论文标题：{{paper_title}}

论文内容：
{{paper_text}}
"""

DEFAULT_USER_TEMPLATE = """请只生成当前这一层内容，不要输出其他层。

输出约束：
- 本层标题必须是：`## {{layer_index}}) {{layer_title}}`
- 当 `{{layer_index}}` 为 `1` 时，必须在本层标题前先输出：`# <insight导向标题>`、`> 标题：{{paper_title}}`、`> 作者：...`、`> 单位：...`、`> 发布信息：...`
- 本层结束后不要追加下一层标题
- 若一个观点可用一句话讲清，则只写一句，不要同义重复
- 原则一（清晰优先）：优先主谓宾短句；删除“众所周知/毫无疑问”等无贡献连接词
- 使用叙事段落为主，避免列表堆砌
- 长段落优先拆分：段首先写结论句，再空一行补充证据句
- 若段首句是核心观点或结果，用 `**...**` 加粗
- 必须基于论文证据，不确定内容写 `[未在原文找到直接证据]`
- 少用缩写，必要术语首次出现时给简短解释或类比
- 原则二（动态解释）：遇到高抽象词汇（如“鲁棒性/泛化能力”）时，优先补一条机制描述；必要时补一个直观类比或具体例子，并尽量带 Figure/Table/图/表 编号
- 保持观点鲜明但克制
- 不要输出“证据锚点：...”字样
- 不要输出“建议投入/暂缓投入/决策建议”

本层目标：{{layer_focus}}
本层硬性要求：{{layer_requirements}}

叙事蓝图：
{{narrative_plan}}

已生成内容（用于承接上下文）：
{{generated_so_far}}

论文标题：{{paper_title}}

论文内容：
{{paper_text}}
"""

DEFAULT_REVIEW_SYSTEM_PROMPT = (
    "You are a strict readability reviewer for five-layer paper summaries."
)

DEFAULT_REVIEW_TEMPLATE = """请评审下列五层总结，重点看连贯性、可读性、证据充分性、语病和理解阻力。

输出格式：
[REVIEW-VERDICT]
- 连贯性评分: <1-5>
- 可理解性评分: <1-5>
- 结论: 通过/不通过

[MAJOR-ISSUES]
- 至少 3 条关键问题
- 每条问题用标签开头：`需精简`（车轱辘话/重复）或 `需补解释`（抽象词缺语境）

[LANGUAGE-AND-FRICTION]
- 语病问题：列出最影响理解的 1-3 处
- 理解阻力问题：从非专家读者视角列出 1-3 处“读到会卡住”的表达

[LAYER-FIXES]
- Layer 1:
- Layer 2:
- Layer 3:
- Layer 4:
- Layer 5:

论文标题：{{paper_title}}

叙事蓝图：
{{narrative_plan}}

草稿内容：
{{draft_summary}}
"""

DEFAULT_REWRITE_SYSTEM_PROMPT = (
    "You are a senior editor polishing five-layer technical summaries in Chinese."
)

DEFAULT_REWRITE_TEMPLATE = """请根据叙事蓝图和审校意见，对以下五层总结做终稿润色。

要求：
- 保留事实，不补充无依据内容
- 输出开头必须先给出 `# <insight导向标题>`、`> 标题：{{paper_title}}`、`> 作者：...`、`> 单位：...`、`> 发布信息：...`
- 强化层间衔接
- 保持五层标题与 `---` 分割线结构
- 提升表达清晰度和传播性
- 长段落优先拆分为“结论句 + 空行 + 证据句”
- 核心结论句可用 `**...**` 加粗
- 删除“证据锚点：...”等内部过程性文本
- 不输出“建议投入/暂缓投入/决策建议”段落
- 信息融合：把审校补充的解释自然揉进原段，不要出现“解释：...”式割裂格式
- 紧凑化处理：在不丢失信息的前提下，删除不推动逻辑的形容词/副词和重复表述
- 若一个观点可用一句话讲清，则保持一句；遇到抽象词时补机制描述，必要时补直观例子

论文标题：{{paper_title}}

叙事蓝图：
{{narrative_plan}}

审校意见：
{{editor_feedback}}

草稿内容：
{{draft_summary}}
"""

_RISKY_TERM_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\breward hacking\b", re.IGNORECASE), "reward gaming"),
    (re.compile(r"\bhacking\b", re.IGNORECASE), "strategic gaming"),
    (re.compile(r"\bhack\b", re.IGNORECASE), "gaming"),
    (re.compile(r"\battacks?\b", re.IGNORECASE), "critiques"),
    (re.compile(r"\bvulnerabilities\b", re.IGNORECASE), "weaknesses"),
    (re.compile(r"\bvulnerability\b", re.IGNORECASE), "weakness"),
    (re.compile(r"\bjailbreak(?:s|ing)?\b", re.IGNORECASE), "policy bypass"),
    (re.compile(r"\bexploit(?:s|ed|ing)?\b", re.IGNORECASE), "misuse"),
)


def _normalize_paper_text_for_prompt(paper_text: str) -> str:
    normalized = paper_text
    for pattern, replacement in _RISKY_TERM_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _replace_placeholders(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def load_prompt_template(path: Path | None) -> str:
    if path is None:
        return DEFAULT_USER_TEMPLATE
    return path.read_text(encoding="utf-8")


def get_five_layer_specs() -> list[dict[str, str]]:
    return [dict(item) for item in FIVE_LAYER_SPECS]


def build_story_planner_prompt(template: str, paper_title: str, paper_text: str) -> str:
    safe_paper_text = _normalize_paper_text_for_prompt(paper_text)
    if "{{paper_text}}" in template or "{{paper_title}}" in template:
        return _replace_placeholders(
            template,
            {
                "{{paper_title}}": paper_title,
                "{{paper_text}}": safe_paper_text,
            },
        )

    return f"{template.strip()}\n\n论文标题：{paper_title}\n\n论文内容：\n{safe_paper_text}"


def build_layer_prompt(
    template: str,
    paper_title: str,
    paper_text: str,
    narrative_plan: str,
    layer_index: str,
    layer_title: str,
    layer_focus: str,
    layer_requirements: str,
    generated_so_far: str,
) -> str:
    safe_plan = narrative_plan.strip() or "[规划代理未产出结果]"
    safe_prefix = generated_so_far.strip() or "[当前为第一层，无前置内容]"
    safe_paper_text = _normalize_paper_text_for_prompt(paper_text)

    placeholders = {
        "{{paper_title}}": paper_title,
        "{{paper_text}}": safe_paper_text,
        "{{narrative_plan}}": safe_plan,
        "{{layer_index}}": layer_index,
        "{{layer_title}}": layer_title,
        "{{layer_focus}}": layer_focus,
        "{{layer_requirements}}": layer_requirements,
        "{{generated_so_far}}": safe_prefix,
    }

    if "{{paper_text}}" in template or "{{paper_title}}" in template:
        rendered = _replace_placeholders(template, placeholders)
        if "{{generated_so_far}}" not in template:
            rendered = f"{rendered}\n\n已生成内容：\n{safe_prefix}"
        return rendered

    return (
        f"{template.strip()}\n\n"
        f"当前层：## {layer_index}) {layer_title}\n"
        f"本层目标：{layer_focus}\n"
        f"本层硬性要求：{layer_requirements}\n\n"
        f"论文标题：{paper_title}\n\n"
        f"叙事蓝图：\n{safe_plan}\n\n"
        f"已生成内容：\n{safe_prefix}\n\n"
        f"论文内容：\n{safe_paper_text}"
    )


def build_user_prompt(
    template: str,
    paper_title: str,
    paper_text: str,
    narrative_plan: str = "",
) -> str:
    """Backward-compatible helper. Generates the first layer prompt."""
    first = FIVE_LAYER_SPECS[0]
    return build_layer_prompt(
        template=template,
        paper_title=paper_title,
        paper_text=paper_text,
        narrative_plan=narrative_plan,
        layer_index=first["index"],
        layer_title=first["title"],
        layer_focus=first["focus"],
        layer_requirements=first["requirements"],
        generated_so_far="",
    )


def build_review_prompt(
    template: str,
    paper_title: str,
    narrative_plan: str,
    draft_summary: str,
) -> str:
    safe_plan = narrative_plan.strip() or "[规划代理未产出结果]"
    if "{{draft_summary}}" in template or "{{paper_title}}" in template:
        rendered = _replace_placeholders(
            template,
            {
                "{{paper_title}}": paper_title,
                "{{draft_summary}}": draft_summary,
                "{{narrative_plan}}": safe_plan,
            },
        )
        if "{{narrative_plan}}" not in template:
            return f"{rendered}\n\n叙事蓝图：\n{safe_plan}"
        return rendered

    return f"{template.strip()}\n\n论文标题：{paper_title}\n\n叙事蓝图：\n{safe_plan}\n\n草稿内容：\n{draft_summary}"


def build_rewrite_prompt(
    template: str,
    paper_title: str,
    draft_summary: str,
    narrative_plan: str = "",
    editor_feedback: str = "",
) -> str:
    safe_plan = narrative_plan.strip() or "[规划代理未产出结果]"
    safe_feedback = editor_feedback.strip() or "[审校代理未产出结果]"
    if "{{draft_summary}}" in template or "{{paper_title}}" in template:
        rendered = _replace_placeholders(
            template,
            {
                "{{paper_title}}": paper_title,
                "{{draft_summary}}": draft_summary,
                "{{narrative_plan}}": safe_plan,
                "{{editor_feedback}}": safe_feedback,
            },
        )
        if "{{narrative_plan}}" not in template:
            rendered = f"{rendered}\n\n叙事蓝图：\n{safe_plan}"
        if "{{editor_feedback}}" not in template:
            rendered = f"{rendered}\n\n编辑审校意见：\n{safe_feedback}"
        return rendered

    return f"{template.strip()}\n\n论文标题：{paper_title}\n\n叙事蓝图：\n{safe_plan}\n\n编辑审校意见：\n{safe_feedback}\n\n草稿内容：\n{draft_summary}"
