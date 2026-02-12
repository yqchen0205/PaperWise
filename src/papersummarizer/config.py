"""Environment-based configuration for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .exceptions import ConfigError
from .models import MinerUParseOptions


@dataclass(frozen=True)
class Settings:
    mineru_api_token: str
    mineru_base_url: str
    mineru_timeout_sec: int
    mineru_poll_interval_sec: float
    mineru_parse_options: MinerUParseOptions

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_timeout_sec: int
    summary_story_planning_enabled: bool
    summary_review_enabled: bool
    summary_rewrite_enabled: bool
    summary_layered_generation_enabled: bool
    summary_token_usage_enabled: bool
    summary_format: str

    network_trust_env: bool

    output_dir: Path
    prompt_template_path: Path | None


_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _read_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return default


def _read_bool(*keys: str, default: bool) -> bool:
    raw = _read_env(*keys)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def _read_int(*keys: str, default: int) -> int:
    raw = _read_env(*keys)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer for {keys[0]}: {raw}") from exc


def _read_float(*keys: str, default: float) -> float:
    raw = _read_env(*keys)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid float for {keys[0]}: {raw}") from exc


def load_settings(dotenv_path: str | Path | None = None) -> Settings:
    """Load project settings from .env and OS env vars."""

    load_dotenv(dotenv_path=dotenv_path, override=False)

    mineru_api_token = _read_env("MINERU_API_TOKEN", "MINERU_API_KEY")
    openai_api_key = _read_env("OPENAI_API_KEY", "API_KEY")

    missing: list[str] = []
    if not mineru_api_token:
        missing.append("MINERU_API_TOKEN")
    if not openai_api_key:
        missing.append("OPENAI_API_KEY (or API_KEY)")

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required environment variables: {joined}")

    prompt_template_raw = _read_env("PROMPT_TEMPLATE_PATH")
    prompt_template_path = Path(prompt_template_raw) if prompt_template_raw else None

    parse_options = MinerUParseOptions(
        enable_formula=_read_bool("MINERU_ENABLE_FORMULA", default=True),
        enable_table=_read_bool("MINERU_ENABLE_TABLE", default=True),
        language=_read_env("MINERU_LANGUAGE", default="en") or "en",
        is_ocr=_read_bool("MINERU_IS_OCR", default=True),
        model_version=_read_env("MINERU_MODEL_VERSION", default="vlm") or "vlm",
    )

    return Settings(
        mineru_api_token=mineru_api_token,
        mineru_base_url=(
            _read_env("MINERU_BASE_URL", default="https://mineru.net")
            or "https://mineru.net"
        ).rstrip("/"),
        mineru_timeout_sec=_read_int("MINERU_TIMEOUT_SEC", default=60),
        mineru_poll_interval_sec=_read_float("MINERU_POLL_INTERVAL_SEC", default=2.0),
        mineru_parse_options=parse_options,
        openai_api_key=openai_api_key,
        openai_base_url=(
            _read_env(
                "OPENAI_BASE_URL", "BASE_URL", default="https://api.openai.com/v1"
            )
            or "https://api.openai.com/v1"
        ).rstrip("/"),
        openai_model=_read_env("OPENAI_MODEL", default="gpt-4.1-mini")
        or "gpt-4.1-mini",
        openai_timeout_sec=_read_int("OPENAI_TIMEOUT_SEC", default=120),
        summary_story_planning_enabled=_read_bool(
            "SUMMARY_STORY_PLANNING_ENABLED", default=True
        ),
        summary_review_enabled=_read_bool("SUMMARY_REVIEW_ENABLED", default=True),
        summary_rewrite_enabled=_read_bool("SUMMARY_REWRITE_ENABLED", default=True),
        summary_layered_generation_enabled=_read_bool(
            "SUMMARY_LAYERED_GENERATION_ENABLED", default=True
        ),
        summary_token_usage_enabled=_read_bool(
            "SUMMARY_TOKEN_USAGE_ENABLED", default=True
        ),
        summary_format=_read_env("SUMMARY_FORMAT", default="five_layers_v1")
        or "five_layers_v1",
        network_trust_env=_read_bool("NETWORK_TRUST_ENV", default=False),
        output_dir=Path(
            _read_env("OUTPUT_DIR", default="outputs/runs/default")
            or "outputs/runs/default"
        ),
        prompt_template_path=prompt_template_path,
    )
