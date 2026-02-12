from types import SimpleNamespace

from papersummarizer.openai_summarizer import OpenAISummarizer


def _response(content: str, usage: dict[str, int] | None = None):
    payload = {
        "choices": [SimpleNamespace(message=SimpleNamespace(content=content))],
        "usage": usage,
    }
    return SimpleNamespace(**payload)


class FakeChatCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(responses=responses)
        )


def test_summarizer_runs_layered_flow_with_usage_tracking():
    responses = [
        _response(
            "story plan",
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        ),
        _response(
            "Layer 1 one-liner",
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        ),
        _response(
            "Layer 2 痛点 研究空白 核心洞察",
            {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        ),
        _response(
            "Layer 3 架构 Figure 1 创新",
            {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
        ),
        _response(
            "Layer 4 SOTA 消融 Figure 2 Table 1 可视化",
            {"prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20},
        ),
        _response(
            "Layer 5 局限 未来方向 建议投入",
            {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
        ),
        _response(
            "review feedback",
            {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        ),
        _response(
            "final summary",
            {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        ),
    ]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="test-model",
        timeout_sec=10,
        prompt_template="{{generated_so_far}}\n\n{{paper_text}}",
        client=fake_client,
    )

    result = summarizer.summarize_with_metrics(
        paper_title="Paper A", paper_text="content"
    )

    assert result.summary_text == "final summary"
    assert len(fake_client.chat.completions.calls) == 8
    assert len(result.token_usage.steps) == 8
    assert result.token_usage.usage_available is True

    usage = result.token_usage.to_dict()
    assert usage["aggregate"]["prompt_tokens"] == 92
    assert usage["aggregate"]["completion_tokens"] == 32
    assert usage["aggregate"]["total_tokens"] == 124
    assert usage["aggregate"]["step_count"] == 8

    first_call = fake_client.chat.completions.calls[0]
    assert first_call["model"] == "test-model"
    assert first_call["temperature"] == 0.1
    assert "Paper A" in first_call["messages"][1]["content"]

    layer_call = fake_client.chat.completions.calls[2]
    assert layer_call["temperature"] == 0.2
    assert "Layer 1 one-liner" in layer_call["messages"][1]["content"]


def test_summarizer_can_disable_rewrite_pass():
    responses = [
        _response("story plan"),
        _response("l1"),
        _response("l2"),
        _response("l3"),
        _response("l4"),
        _response("l5"),
    ]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="test-model",
        timeout_sec=10,
        prompt_template="{{generated_so_far}}\n\n{{paper_text}}",
        rewrite_enabled=False,
        client=fake_client,
    )

    summary = summarizer.summarize(paper_title="Paper A", paper_text="content")

    assert "## 1) TL;DR" in summary
    assert "## 5) Insights & Decision" in summary
    assert len(fake_client.chat.completions.calls) == 6


def test_summarizer_can_disable_story_planning_and_review():
    responses = [
        _response("l1"),
        _response("l2"),
        _response("l3"),
        _response("l4"),
        _response("l5"),
        _response("final summary"),
    ]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="test-model",
        timeout_sec=10,
        prompt_template="{{generated_so_far}}\n\n{{paper_text}}",
        story_planning_enabled=False,
        review_enabled=False,
        client=fake_client,
    )

    summary = summarizer.summarize(paper_title="Paper A", paper_text="content")

    assert summary == "final summary"
    assert len(fake_client.chat.completions.calls) == 6


def test_summarizer_handles_missing_usage_gracefully():
    responses = [
        _response("story plan"),
        _response("l1"),
        _response("l2"),
        _response("l3"),
        _response("l4"),
        _response("l5"),
        _response("review"),
        _response("final"),
    ]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="test-model",
        timeout_sec=10,
        prompt_template="{{generated_so_far}}\n\n{{paper_text}}",
        client=fake_client,
    )

    result = summarizer.summarize_with_metrics(
        paper_title="Paper A", paper_text="content"
    )

    assert result.summary_text == "final"
    assert result.token_usage.enabled is True
    assert result.token_usage.usage_available is False
    assert len(result.token_usage.steps) == 8


def test_summarizer_forces_temperature_for_kimi_k25_models():
    responses = [_response("final summary")]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="kimi-k2.5",
        timeout_sec=10,
        prompt_template="{{generated_so_far}}\n\n{{paper_text}}",
        story_planning_enabled=False,
        layered_generation_enabled=False,
        rewrite_enabled=False,
        client=fake_client,
    )

    summary = summarizer.summarize(paper_title="Paper A", paper_text="content")

    assert summary == "final summary"
    assert fake_client.chat.completions.calls[0]["temperature"] == 1.0


def test_summarizer_trims_references_and_appendix_from_prompt_input():
    responses = [_response("story plan"), _response("full draft")]
    fake_client = FakeClient(responses=responses)
    summarizer = OpenAISummarizer(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model="test-model",
        timeout_sec=10,
        prompt_template="{{paper_text}}",
        story_planning_enabled=True,
        layered_generation_enabled=False,
        rewrite_enabled=False,
        client=fake_client,
    )

    paper_text = """# TITLE
# 1 INTRO
Body content that should be summarized.

# REFERENCES
Ref A

# A APPENDIX
Evaluation:
Scoring:
<score>3</score>
"""

    summarizer.summarize(paper_title="Paper A", paper_text=paper_text)

    first_user_prompt = fake_client.chat.completions.calls[0]["messages"][1]["content"]
    second_user_prompt = fake_client.chat.completions.calls[1]["messages"][1]["content"]

    assert "Body content that should be summarized." in first_user_prompt
    assert "Body content that should be summarized." in second_user_prompt
    assert "# REFERENCES" not in first_user_prompt
    assert "# REFERENCES" not in second_user_prompt
    assert "Scoring:" not in first_user_prompt
    assert "Scoring:" not in second_user_prompt
    assert "<score>3</score>" not in first_user_prompt
    assert "<score>3</score>" not in second_user_prompt
