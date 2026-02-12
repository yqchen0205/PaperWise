"""OpenAI-compatible summarizer client."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import httpx
from openai import OpenAI

from .exceptions import OpenAISummarizationError
from .models import LLMCallUsage, SummarizationResult, SummarizationTokenUsage
from .prompts import (
    DEFAULT_REVIEW_SYSTEM_PROMPT,
    DEFAULT_REVIEW_TEMPLATE,
    DEFAULT_REWRITE_SYSTEM_PROMPT,
    DEFAULT_REWRITE_TEMPLATE,
    DEFAULT_STORY_PLANNER_SYSTEM_PROMPT,
    DEFAULT_STORY_PLANNER_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    build_layer_prompt,
    build_review_prompt,
    build_rewrite_prompt,
    build_story_planner_prompt,
    get_five_layer_specs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LAYER_HEADER_PATTERN = re.compile(r"^\s*##\s*([1-5])\)")
UNIT_TEMPERATURE_MODELS = ("kimi-k2.5",)
MAIN_BODY_CUTOFF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*#\s*(?:\d+[.)]?\s*)?(?:references|bibliography)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*#\s*(?:\d+[.)]?\s*)?(?:appendix|appendices)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*#\s*(?:\d+[.)]?\s*)?(?:supplementary(?:\s+material)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:references|bibliography)\s*$", re.IGNORECASE),
)


class OpenAISummarizer:
    """Generate paper summaries from parsed paper text."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: int,
        prompt_template: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_input_chars: int = 120_000,
        story_planning_enabled: bool = True,
        story_planner_prompt_template: str = DEFAULT_STORY_PLANNER_TEMPLATE,
        story_planner_system_prompt: str = DEFAULT_STORY_PLANNER_SYSTEM_PROMPT,
        rewrite_enabled: bool = True,
        review_enabled: bool = True,
        review_prompt_template: str = DEFAULT_REVIEW_TEMPLATE,
        review_system_prompt: str = DEFAULT_REVIEW_SYSTEM_PROMPT,
        rewrite_prompt_template: str = DEFAULT_REWRITE_TEMPLATE,
        rewrite_system_prompt: str = DEFAULT_REWRITE_SYSTEM_PROMPT,
        layered_generation_enabled: bool = True,
        token_usage_enabled: bool = True,
        trust_env: bool = False,
        client: OpenAI | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.max_input_chars = max_input_chars
        self.story_planning_enabled = story_planning_enabled
        self.story_planner_prompt_template = story_planner_prompt_template
        self.story_planner_system_prompt = story_planner_system_prompt
        self.rewrite_enabled = rewrite_enabled
        self.review_enabled = review_enabled
        self.review_prompt_template = review_prompt_template
        self.review_system_prompt = review_system_prompt
        self.rewrite_prompt_template = rewrite_prompt_template
        self.rewrite_system_prompt = rewrite_system_prompt
        self.layered_generation_enabled = layered_generation_enabled
        self.token_usage_enabled = token_usage_enabled
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_sec,
            http_client=httpx.Client(
                timeout=timeout_sec,
                trust_env=trust_env,
            ),
        )
        self.progress_callback = progress_callback

    def summarize(self, paper_title: str, paper_text: str) -> str:
        return self.summarize_with_metrics(
            paper_title=paper_title,
            paper_text=paper_text,
        ).summary_text

    def summarize_with_metrics(
        self,
        paper_title: str,
        paper_text: str,
    ) -> SummarizationResult:
        safe_text = self._prepare_input_text(paper_text)

        usage_steps: list[LLMCallUsage] = []
        narrative_plan = ""

        if self.story_planning_enabled:
            if self.progress_callback:
                self.progress_callback("story_planning", "Generating narrative plan...")

            planner_prompt = build_story_planner_prompt(
                template=self.story_planner_prompt_template,
                paper_title=paper_title,
                paper_text=safe_text,
            )
            narrative_plan, planner_usage = self._summarize_once(
                step_name="story_planner",
                system_prompt=self.story_planner_system_prompt,
                user_prompt=planner_prompt,
                temperature=0.1,
            )
            if self.token_usage_enabled:
                usage_steps.append(planner_usage)

            if self.progress_callback:
                total_tokens = planner_usage.total_tokens or 0
                self.progress_callback("story_planning", f"✓ ({total_tokens} tokens)")

        draft_summary, draft_usage = self._build_draft_summary(
            paper_title=paper_title,
            paper_text=safe_text,
            narrative_plan=narrative_plan,
        )
        if self.token_usage_enabled:
            usage_steps.extend(draft_usage)

        final_summary = draft_summary
        if self.rewrite_enabled:
            review_feedback = ""
            if self.review_enabled:
                if self.progress_callback:
                    self.progress_callback("review", "Reviewing draft summary...")

                review_prompt = build_review_prompt(
                    template=self.review_prompt_template,
                    paper_title=paper_title,
                    narrative_plan=narrative_plan,
                    draft_summary=draft_summary,
                )
                review_feedback, review_usage = self._summarize_once(
                    step_name="review",
                    system_prompt=self.review_system_prompt,
                    user_prompt=review_prompt,
                    temperature=0.1,
                )
                if self.token_usage_enabled:
                    usage_steps.append(review_usage)

                if self.progress_callback:
                    total_tokens = review_usage.total_tokens or 0
                    self.progress_callback("review", f"✓ ({total_tokens} tokens)")

            if self.progress_callback:
                self.progress_callback("rewrite", "Rewriting final summary...")

            rewrite_prompt = build_rewrite_prompt(
                template=self.rewrite_prompt_template,
                paper_title=paper_title,
                draft_summary=draft_summary,
                narrative_plan=narrative_plan,
                editor_feedback=review_feedback,
            )
            final_summary, rewrite_usage = self._summarize_once(
                step_name="rewrite",
                system_prompt=self.rewrite_system_prompt,
                user_prompt=rewrite_prompt,
                temperature=0.1,
            )
            if self.token_usage_enabled:
                usage_steps.append(rewrite_usage)

            if self.progress_callback:
                total_tokens = rewrite_usage.total_tokens or 0
                self.progress_callback("rewrite", f"✓ ({total_tokens} tokens)")

        usage = SummarizationTokenUsage(
            enabled=self.token_usage_enabled,
            usage_available=(
                self.token_usage_enabled
                and bool(usage_steps)
                and all(step.usage_available for step in usage_steps)
            ),
            steps=usage_steps,
        )
        return SummarizationResult(
            summary_text=final_summary.strip(), token_usage=usage
        )

    def _prepare_input_text(self, paper_text: str) -> str:
        safe_text = self._extract_main_body_text(paper_text)
        if len(safe_text) > self.max_input_chars:
            safe_text = (
                safe_text[: self.max_input_chars]
                + "\n\n[Truncated: input text exceeds max_input_chars]"
            )
        return safe_text

    def _extract_main_body_text(self, paper_text: str) -> str:
        lines = paper_text.splitlines()
        cutoff_index: int | None = None

        for index, line in enumerate(lines):
            if any(pattern.match(line) for pattern in MAIN_BODY_CUTOFF_PATTERNS):
                cutoff_index = index
                break

        if cutoff_index is None:
            return paper_text

        main_body = "\n".join(lines[:cutoff_index]).strip()
        if not main_body:
            return paper_text
        return main_body

    def _build_draft_summary(
        self,
        paper_title: str,
        paper_text: str,
        narrative_plan: str,
    ) -> tuple[str, list[LLMCallUsage]]:
        if not self.layered_generation_enabled:
            prompt = build_layer_prompt(
                template=self.prompt_template,
                paper_title=paper_title,
                paper_text=paper_text,
                narrative_plan=narrative_plan,
                layer_index="1",
                layer_title="TL;DR",
                layer_focus="输出完整五层总结",
                layer_requirements="一次生成完整五层结构，每层之间使用 --- 分割",
                generated_so_far="",
            )
            summary, usage = self._summarize_once(
                step_name="draft_full",
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                temperature=0.2,
            )
            return summary, [usage]

        layer_outputs: list[str] = []
        usage_steps: list[LLMCallUsage] = []

        layer_specs = get_five_layer_specs()
        for idx, layer_spec in enumerate(layer_specs, 1):
            layer_key = f"layer_{idx}"
            layer_name = layer_spec["title"]

            if self.progress_callback:
                self.progress_callback(layer_key, f"Generating {layer_name}...")

            user_prompt = build_layer_prompt(
                template=self.prompt_template,
                paper_title=paper_title,
                paper_text=paper_text,
                narrative_plan=narrative_plan,
                layer_index=layer_spec["index"],
                layer_title=layer_spec["title"],
                layer_focus=layer_spec["focus"],
                layer_requirements=layer_spec["requirements"],
                generated_so_far=self._join_layers(layer_outputs),
            )
            layer_text, layer_usage = self._summarize_once(
                step_name=f"layer_{layer_spec['index']}_{layer_spec['key']}",
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            )
            usage_steps.append(layer_usage)
            layer_outputs.append(
                self._normalize_layer_block(
                    layer_text=layer_text,
                    layer_index=layer_spec["index"],
                    layer_title=layer_spec["title"],
                )
            )

            if self.progress_callback:
                total_tokens = layer_usage.total_tokens or 0
                self.progress_callback(layer_key, f"✓ {layer_name} ({total_tokens} tokens)")

        return self._join_layers(layer_outputs), usage_steps

    def _join_layers(self, layers: list[str]) -> str:
        if not layers:
            return ""
        return "\n\n---\n\n".join(layer.strip() for layer in layers if layer.strip())

    def _normalize_layer_block(
        self,
        layer_text: str,
        layer_index: str,
        layer_title: str,
    ) -> str:
        stripped = layer_text.strip()
        if not stripped:
            return f"## {layer_index}) {layer_title}\n\n[未生成有效内容]"

        if LAYER_HEADER_PATTERN.search(stripped):
            return stripped

        return f"## {layer_index}) {layer_title}\n\n{stripped}"

    def _summarize_once(
        self,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> tuple[str, LLMCallUsage]:
        response = self._request_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        if not response.choices:
            raise OpenAISummarizationError("OpenAI returned no choices")

        content = self._extract_content(response.choices[0].message.content)
        if not content:
            raise OpenAISummarizationError("OpenAI returned empty summary")

        usage = self._extract_usage(step_name=step_name, response=response)
        return content.strip(), usage

    def _request_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._resolve_temperature(temperature),
            )
        except Exception as exc:
            raise OpenAISummarizationError(f"OpenAI request failed: {exc}") from exc

    def _resolve_temperature(self, requested_temperature: float) -> float:
        normalized_model = self.model.strip().lower()
        for fixed_model in UNIT_TEMPERATURE_MODELS:
            if normalized_model == fixed_model or normalized_model.startswith(
                f"{fixed_model}-"
            ):
                return 1.0
        return requested_temperature

    def _extract_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
            return "\n".join(chunks)

        return ""

    def _extract_usage(self, step_name: str, response: Any) -> LLMCallUsage:
        if not self.token_usage_enabled:
            return LLMCallUsage(
                step_name=step_name,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                usage_available=False,
            )

        usage = getattr(response, "usage", None)
        if usage is None:
            return LLMCallUsage(
                step_name=step_name,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                usage_available=False,
            )

        prompt_tokens = self._read_usage_value(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._read_usage_value(
            usage, "completion_tokens", "output_tokens"
        )
        total_tokens = self._read_usage_value(usage, "total_tokens")

        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        usage_available = (
            prompt_tokens is not None
            or completion_tokens is not None
            or total_tokens is not None
        )
        return LLMCallUsage(
            step_name=step_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage_available=usage_available,
        )

    def _read_usage_value(self, usage: Any, *keys: str) -> int | None:
        for key in keys:
            value = None
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)

            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None
