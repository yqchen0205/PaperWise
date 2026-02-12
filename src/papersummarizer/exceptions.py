"""Custom exceptions for the paper summarizer."""


class PaperSummarizerError(Exception):
    """Base exception for the project."""


class ConfigError(PaperSummarizerError):
    """Raised when required configuration is missing or invalid."""


class MinerUApiError(PaperSummarizerError):
    """Raised when MinerU API call fails."""


class MinerUParseError(PaperSummarizerError):
    """Raised when MinerU parse workflow fails."""


class OpenAISummarizationError(PaperSummarizerError):
    """Raised when LLM summarization fails."""
