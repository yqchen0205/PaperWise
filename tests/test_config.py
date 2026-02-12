from pathlib import Path

from papersummarizer.config import load_settings


ENV_KEYS = [
    "MINERU_API_TOKEN",
    "MINERU_API_KEY",
    "MINERU_BASE_URL",
    "MINERU_TIMEOUT_SEC",
    "MINERU_POLL_INTERVAL_SEC",
    "MINERU_ENABLE_FORMULA",
    "MINERU_ENABLE_TABLE",
    "MINERU_LANGUAGE",
    "MINERU_IS_OCR",
    "MINERU_MODEL_VERSION",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_TIMEOUT_SEC",
    "SUMMARY_STORY_PLANNING_ENABLED",
    "SUMMARY_REVIEW_ENABLED",
    "SUMMARY_REWRITE_ENABLED",
    "SUMMARY_LAYERED_GENERATION_ENABLED",
    "SUMMARY_TOKEN_USAGE_ENABLED",
    "SUMMARY_FORMAT",
    "API_KEY",
    "BASE_URL",
    "OUTPUT_DIR",
    "PROMPT_TEMPLATE_PATH",
]


def clear_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_settings_from_dotenv_with_aliases(tmp_path, monkeypatch):
    clear_env(monkeypatch)

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "MINERU_API_TOKEN=test-mineru-token",
                "MINERU_BASE_URL=https://mineru.example.com",
                "API_KEY=test-openai-key",
                "BASE_URL=https://openai.example.com/v1",
                "OPENAI_MODEL=test-model",
                "OUTPUT_DIR=custom_outputs",
                "PROMPT_TEMPLATE_PATH=prompt.md",
                "SUMMARY_STORY_PLANNING_ENABLED=false",
                "SUMMARY_REVIEW_ENABLED=false",
                "SUMMARY_REWRITE_ENABLED=false",
                "SUMMARY_LAYERED_GENERATION_ENABLED=false",
                "SUMMARY_TOKEN_USAGE_ENABLED=false",
                "SUMMARY_FORMAT=five_layers_v1",
                "MINERU_MODEL_VERSION=test-model-version",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(dotenv_path=dotenv)

    assert settings.mineru_api_token == "test-mineru-token"
    assert settings.mineru_base_url == "https://mineru.example.com"
    assert settings.openai_api_key == "test-openai-key"
    assert settings.openai_base_url == "https://openai.example.com/v1"
    assert settings.openai_model == "test-model"
    assert settings.output_dir == Path("custom_outputs")
    assert settings.prompt_template_path == Path("prompt.md")
    assert settings.mineru_parse_options.model_version == "test-model-version"
    assert settings.summary_story_planning_enabled is False
    assert settings.summary_review_enabled is False
    assert settings.summary_rewrite_enabled is False
    assert settings.summary_layered_generation_enabled is False
    assert settings.summary_token_usage_enabled is False
    assert settings.summary_format == "five_layers_v1"


def test_load_settings_defaults(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("MINERU_API_TOKEN", "mineru")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    settings = load_settings(dotenv_path=tmp_path / "missing.env")

    assert settings.mineru_base_url == "https://mineru.net"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.output_dir == Path("outputs/runs/default")
    assert settings.mineru_parse_options.model_version == "vlm"
    assert settings.summary_story_planning_enabled is True
    assert settings.summary_review_enabled is True
    assert settings.summary_rewrite_enabled is True
    assert settings.summary_layered_generation_enabled is True
    assert settings.summary_token_usage_enabled is True
    assert settings.summary_format == "five_layers_v1"
