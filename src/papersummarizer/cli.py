"""CLI entrypoint for paper summarization pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .mineru_client import MinerUClient
from .mineru_parser import MinerUPdfParser
from .openai_summarizer import OpenAISummarizer
from .pipeline import PaperSummarizationPipeline
from .progress import RichProgressTracker, track_processing
from .prompts import load_prompt_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDFs via MinerU and summarize via OpenAI"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="PDF file or directory containing PDFs",
    )
    parser.add_argument("--dotenv", type=Path, default=None, help="Path to .env file")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override OUTPUT_DIR"
    )
    parser.add_argument("--model", type=str, default=None, help="Override OPENAI_MODEL")
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        help="Markdown prompt template path; supports {{paper_title}} and {{paper_text}}",
    )
    parser.add_argument(
        "--summary-format",
        type=str,
        default=None,
        help="Summary format contract (default: five_layers_v1)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Process only the first N PDFs"
    )
    parser.add_argument(
        "--pdf-url",
        action="append",
        default=None,
        help="Remote PDF URL. Repeat this argument to process multiple URLs.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if summary output already exists",
    )
    parser.add_argument(
        "--poll-timeout-sec",
        type=int,
        default=900,
        help="MinerU polling timeout for each file",
    )
    parser.add_argument(
        "--layered-generation",
        action="store_true",
        default=None,
        help="Force enable layered generation",
    )
    parser.add_argument(
        "--no-layered-generation",
        action="store_true",
        default=False,
        help="Force disable layered generation",
    )
    parser.add_argument(
        "--token-usage",
        action="store_true",
        default=None,
        help="Force enable token usage tracking",
    )
    parser.add_argument(
        "--no-token-usage",
        action="store_true",
        default=False,
        help="Force disable token usage tracking",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        default=False,
        help="Disable Rich progress display, use simple tqdm instead",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = load_settings(dotenv_path=args.dotenv)

    input_path = args.input
    pdf_urls = args.pdf_url or []
    if input_path is None and not pdf_urls:
        raise ValueError("Provide either local input path or at least one --pdf-url")
    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_dir = args.output_dir or settings.output_dir
    model = args.model or settings.openai_model
    summary_format = args.summary_format or settings.summary_format

    layered_generation_enabled = settings.summary_layered_generation_enabled
    if args.layered_generation is True:
        layered_generation_enabled = True
    if args.no_layered_generation:
        layered_generation_enabled = False

    token_usage_enabled = settings.summary_token_usage_enabled
    if args.token_usage is True:
        token_usage_enabled = True
    if args.no_token_usage:
        token_usage_enabled = False

    prompt_path = args.prompt_template or settings.prompt_template_path
    prompt_template = load_prompt_template(prompt_path)

    # Initialize Rich progress tracker (if not disabled)
    progress_tracker = None
    if not args.no_rich:
        try:
            progress_tracker = RichProgressTracker()
            progress_tracker.start()
        except Exception:
            # Fall back to no progress tracking if Rich fails
            progress_tracker = None

    try:
        mineru_client = MinerUClient(
            base_url=settings.mineru_base_url,
            api_token=settings.mineru_api_token,
            timeout_sec=settings.mineru_timeout_sec,
            trust_env=settings.network_trust_env,
        )
        parser = MinerUPdfParser(
            client=mineru_client,
            parse_options=settings.mineru_parse_options,
            poll_interval_sec=settings.mineru_poll_interval_sec,
            poll_timeout_sec=args.poll_timeout_sec,
        )
        summarizer = OpenAISummarizer(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model,
            timeout_sec=settings.openai_timeout_sec,
            prompt_template=prompt_template,
            story_planning_enabled=settings.summary_story_planning_enabled,
            review_enabled=settings.summary_review_enabled,
            rewrite_enabled=settings.summary_rewrite_enabled,
            layered_generation_enabled=layered_generation_enabled,
            token_usage_enabled=token_usage_enabled,
            trust_env=settings.network_trust_env,
        )

        pipeline = PaperSummarizationPipeline(
            parser=parser,
            summarizer=summarizer,
            output_dir=output_dir,
            summary_format=summary_format,
            progress_tracker=progress_tracker,
        )

        results = []
        if input_path is not None:
            results.extend(
                pipeline.run(
                    input_path=input_path,
                    max_files=args.max_files,
                    skip_existing=not args.no_skip_existing,
                )
            )
        if pdf_urls:
            results.extend(
                pipeline.run_urls(
                    pdf_urls=pdf_urls,
                    skip_existing=not args.no_skip_existing,
                )
            )

        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count

        print(f"Finished. total={len(results)} success={success_count} failed={fail_count}")
        if fail_count:
            print("Failed files:")
            for result in results:
                if not result.success:
                    print(f"- {result.pdf_path}: {result.error}")
    finally:
        if progress_tracker:
            progress_tracker.stop()


if __name__ == "__main__":
    main()
