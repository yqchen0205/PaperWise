from papersummarizer.prompts import (
    DEFAULT_REVIEW_TEMPLATE,
    DEFAULT_REWRITE_TEMPLATE,
    DEFAULT_STORY_PLANNER_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    build_layer_prompt,
    build_story_planner_prompt,
)


def test_story_planner_prompt_normalizes_high_risk_terms_in_paper_text():
    template = "标题：{{paper_title}}\n\n正文：{{paper_text}}"
    paper_text = (
        "This section discusses reward hacking and attack behaviors. "
        "Some teams also mention hacking patterns and jailbreak attempts."
    )

    rendered = build_story_planner_prompt(
        template=template,
        paper_title="Test Paper",
        paper_text=paper_text,
    )

    assert "reward hacking" not in rendered.lower()
    assert " attack " not in rendered.lower()
    assert " hacking " not in rendered.lower()
    assert "jailbreak" not in rendered.lower()
    assert "reward gaming" in rendered.lower()
    assert "critique" in rendered.lower()
    assert "policy bypass" in rendered.lower()


def test_layer_prompt_normalizes_high_risk_terms_in_paper_text():
    template = "{{paper_title}}\n{{paper_text}}\n{{generated_so_far}}"
    paper_text = "The benchmark contains attack traces and vulnerability notes."

    rendered = build_layer_prompt(
        template=template,
        paper_title="Paper B",
        paper_text=paper_text,
        narrative_plan="plan",
        layer_index="1",
        layer_title="TL;DR",
        layer_focus="focus",
        layer_requirements="requirements",
        generated_so_far="",
    )

    assert "attack" not in rendered.lower()
    assert "vulnerability" not in rendered.lower()
    assert "critique" in rendered.lower()
    assert "weakness" in rendered.lower()


def test_default_story_planner_template_emphasizes_dedup_and_explanation_slots():
    assert "逻辑去重" in DEFAULT_STORY_PLANNER_TEMPLATE
    assert "解释预埋" in DEFAULT_STORY_PLANNER_TEMPLATE


def test_default_layer_template_emphasizes_concise_and_grounded_explanations():
    assert "若一个观点可用一句话讲清" in DEFAULT_USER_TEMPLATE
    assert "高抽象词汇" in DEFAULT_USER_TEMPLATE
    assert "机制描述" in DEFAULT_USER_TEMPLATE
    assert "直观类比" in DEFAULT_USER_TEMPLATE


def test_default_review_template_checks_language_and_friction():
    assert "语病" in DEFAULT_REVIEW_TEMPLATE
    assert "理解阻力" in DEFAULT_REVIEW_TEMPLATE
    assert "需精简" in DEFAULT_REVIEW_TEMPLATE
    assert "需补解释" in DEFAULT_REVIEW_TEMPLATE


def test_default_rewrite_template_requires_compact_natural_integration():
    assert "信息融合" in DEFAULT_REWRITE_TEMPLATE
    assert "紧凑化处理" in DEFAULT_REWRITE_TEMPLATE
