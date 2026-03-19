#!/usr/bin/env python3
"""Main orchestrator for PDF -> dialogue -> TTS podcast pipeline."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

from llm_writer import (
    LLMWriterError,
    choose_paper_name,
    estimate_tokens_from_text,
    extract_pdf_text,
    generate_dialogue_file_from_text,
    load_api_keys,
)
from tts_audio import TTSAudioError, synthesize_audio_from_dialogue

# ---------------------------------------------------------------------------
# Quick-start config
# ---------------------------------------------------------------------------
# Options:
# - DETAIL_LEVEL: "Default" or "High"
# - Company: "OpenAI" or "Claude"
# - Model examples:
#   - Claude: "claude-3-7-sonnet-latest", "claude-opus-4-6", "claude-sonnet-4-6"
#   - OpenAI: "gpt-4o", "gpt-4o-latest"
# - MODEL_MAX_OUTPUT_TOKENS and THINKING_BUDGET_TOKENS are computed dynamically from context.
#   You can still override at runtime using CLI flags.
# - EXTENDED_THINKING_ALWAYS_ON keeps reasoning mode enabled for all runs.

File = "Performance Analysis of Cell-Free Massive MIMO System With Limited Fronthaul Capacity and Hardware Impairments.pdf"
DETAIL_LEVEL = "High"
Company = "Claude"
Model = "claude-opus-4-6"
API_WRITER_FILE = "apit.txt"
API_VOICE_FILE = "apiv.txt"
FINAL_ROOT = "Final"
EXTENDED_THINKING_ALWAYS_ON = True

# TTS settings
TTS_MODEL = "gpt-4o-mini-tts"
HOST_A_VOICE = "echo"
HOST_B_VOICE = "sage"

SUPPORTED_MODELS = {
    "claude-3-7-sonnet-latest": {
        "provider": "anthropic",
        "max_context_tokens": 200_000,
        "max_output_tokens": 128_000,
        "supports_extended_thinking": True,
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "max_context_tokens": 200_000,
        "max_output_tokens": 128_000,
        "supports_extended_thinking": True,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "max_context_tokens": 200_000,
        "max_output_tokens": 128_000,
        "supports_extended_thinking": True,
    },
    "gpt-4o": {
        "provider": "openai",
        "max_context_tokens": 128_000,
        "max_output_tokens": 16_384,
        "supports_extended_thinking": True,
    },
    "gpt-4o-latest": {
        "provider": "openai",
        "max_context_tokens": 128_000,
        "max_output_tokens": 16_384,
        "supports_extended_thinking": True,
    },
}


class MainError(RuntimeError):
    """Raised for main pipeline configuration/runtime failures."""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the orchestrator."""
    parser = argparse.ArgumentParser(description="Generate podcast assets from a research paper PDF.")
    parser.add_argument("--pdf", default=File, help="Input PDF path.")
    parser.add_argument("--detail-level", choices=["Default", "High"], default=DETAIL_LEVEL)
    parser.add_argument("--company", default=Company, help="OpenAI or Claude")
    parser.add_argument("--model", default=Model, help="Writer model id")
    parser.add_argument("--api-writer-file", default=API_WRITER_FILE, help="Path to writer API keys file (apit.txt)")
    parser.add_argument("--api-voice-file", default=API_VOICE_FILE, help="Path to voice API keys file (apiv.txt)")
    parser.add_argument("--final-root", default=FINAL_ROOT, help="Master output root folder")
    parser.add_argument("--llm-base-url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dialogue-turns", type=int, default=48)
    parser.add_argument("--max-input-chars", type=int, default=300000)
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Manual override")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Manual override")

    parser.add_argument("--tts-model", default=TTS_MODEL)
    parser.add_argument("--host-a-voice", default=HOST_A_VOICE)
    parser.add_argument("--host-b-voice", default=HOST_B_VOICE)
    parser.add_argument("--skip-tts", action="store_true")
    return parser.parse_args()


def _resolve_provider(company: str) -> str:
    value = (company or "").strip().lower()
    if "claude" in value or "anthropic" in value:
        return "anthropic"
    if "openai" in value or "gpt" in value or "chatgpt" in value:
        return "openai"
    raise MainError("Company must be OpenAI or Claude.")


def _resolve_pdf_path(pdf_value: str, root_dir: Path) -> Path:
    if not pdf_value:
        raise MainError("Missing PDF path. Set File or use --pdf.")
    candidate = Path(pdf_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    direct = candidate.resolve()
    if direct.exists():
        return direct

    local = (root_dir / candidate).resolve()
    if local.exists():
        return local
    return local


def _normalize_model_for_provider(provider: str, model: str) -> str:
    m = (model or "").strip()
    lowered = m.lower()
    if provider == "anthropic":
        if lowered in {"opus", "opus 4.6", "claude opus", "claude-opus"}:
            return "claude-opus-4-6"
        if lowered in {"sonnet", "claude sonnet", "claude-sonnet"}:
            return "claude-3-7-sonnet-latest"
    return m


def _lookup_model_profile(model_name: str, provider: str) -> Dict[str, object]:
    for key, value in SUPPORTED_MODELS.items():
        if key.lower() == model_name.lower():
            return value

    fallback_context = 200_000 if provider == "anthropic" else 128_000
    fallback_output_cap = 128_000 if provider == "anthropic" else 16_384
    return {
        "provider": provider,
        "max_context_tokens": fallback_context,
        "max_output_tokens": fallback_output_cap,
        "supports_extended_thinking": True,
    }


def estimate_input_tokens(text: str, model_name: str) -> int:
    """Estimate input token count for adaptive budgeting."""
    return estimate_tokens_from_text(text, model_name)


def calculate_dynamic_budgets(
    *,
    input_tokens: int,
    context_window: int,
    max_output_cap: int,
    detail_level: str,
    supports_extended_thinking: bool,
) -> Tuple[int, int]:
    """Compute max_output_tokens and thinking_budget from context remaining."""
    safety_margin = max(1500, int(context_window * 0.03))
    available = max(1024, context_window - input_tokens - safety_margin)
    available = min(available, max_output_cap)

    detail = detail_level.strip().lower()
    if detail == "high":
        max_output_tokens = available
        thinking_budget = available if supports_extended_thinking else 0
    else:
        sensible_limit = min(8192, available)
        max_output_tokens = max(1024, sensible_limit)
        thinking_budget = min(max_output_tokens // 2, 8000) if supports_extended_thinking else 0

    return max_output_tokens, thinking_budget


def _build_bootstrap_logger() -> logging.Logger:
    logger = logging.getLogger("podcast_bootstrap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def setup_logger(log_path: Path) -> logging.Logger:
    """Create run logger that writes to console and processing.log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_suite")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def run() -> int:
    """Execute full pipeline."""
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    try:
        pdf_path = _resolve_pdf_path(args.pdf, root_dir)
        api_writer_file = Path(args.api_writer_file).expanduser().resolve()
        api_voice_file = Path(args.api_voice_file).expanduser().resolve()
        final_root = Path(args.final_root).expanduser()
        if not final_root.is_absolute():
            final_root = (root_dir / final_root).resolve()
        final_root.mkdir(parents=True, exist_ok=True)

        provider = _resolve_provider(args.company)
        model = _normalize_model_for_provider(provider, args.model)

        bootstrap_logger = _build_bootstrap_logger()
        title, text = extract_pdf_text(pdf_path, bootstrap_logger)
        paper_name = choose_paper_name(pdf_path, title, text)

        paper_dir = final_root / paper_name
        paper_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(paper_dir / "processing.log")

        logger.info("Run started for paper: %s", paper_name)
        logger.info("PDF path: %s", pdf_path)
        logger.info("Output folder: %s", paper_dir)

        copied_pdf = paper_dir / pdf_path.name
        if copied_pdf.resolve() != pdf_path.resolve():
            shutil.copy2(pdf_path, copied_pdf)
            logger.info("Copied source PDF to: %s", copied_pdf)

        writer_api_keys = load_api_keys(api_writer_file)
        voice_api_keys = load_api_keys(api_voice_file)
        logger.info("Writer API file: %s", api_writer_file)
        logger.info("Voice API file: %s", api_voice_file)
        logger.info("Loaded writer API services: %s", ", ".join(sorted(writer_api_keys.keys())))
        logger.info("Loaded voice API services: %s", ", ".join(sorted(voice_api_keys.keys())))

        profile = _lookup_model_profile(model, provider)
        context_window = int(profile["max_context_tokens"])
        model_max_output_cap = int(profile["max_output_tokens"])
        supports_thinking = bool(profile["supports_extended_thinking"]) or EXTENDED_THINKING_ALWAYS_ON

        input_tokens = estimate_input_tokens(text, model)
        dyn_output, dyn_thinking = calculate_dynamic_budgets(
            input_tokens=input_tokens,
            context_window=context_window,
            max_output_cap=model_max_output_cap,
            detail_level=args.detail_level,
            supports_extended_thinking=supports_thinking,
        )

        max_output_tokens = args.max_output_tokens if args.max_output_tokens is not None else dyn_output
        max_output_tokens = min(max_output_tokens, model_max_output_cap)
        thinking_budget = args.thinking_budget if args.thinking_budget is not None else dyn_thinking
        if supports_thinking:
            if thinking_budget >= max_output_tokens:
                thinking_budget = max(0, max_output_tokens - 1)
            if 0 < thinking_budget < 1024:
                thinking_budget = 0
            if EXTENDED_THINKING_ALWAYS_ON and thinking_budget <= 0 and max_output_tokens > 1024:
                thinking_budget = max(1024, max_output_tokens // 2)
                if thinking_budget >= max_output_tokens:
                    thinking_budget = max_output_tokens - 1
        else:
            thinking_budget = 0

        logger.info(
            "LLM config: provider=%s model=%s detail=%s input_tokens~%d context=%d model_output_cap=%d max_output=%d thinking_budget=%d",
            provider,
            model,
            args.detail_level,
            input_tokens,
            context_window,
            model_max_output_cap,
            max_output_tokens,
            thinking_budget,
        )

        dialogue_path = generate_dialogue_file_from_text(
            paper_name=paper_name,
            paper_text=text,
            output_dir=paper_dir,
            api_keys=writer_api_keys,
            llm_provider=provider,
            llm_model=model,
            llm_base_url=args.llm_base_url,
            detail_level=args.detail_level,
            max_output_tokens=max_output_tokens,
            thinking_budget_tokens=thinking_budget,
            dialogue_turns=max(12, int(args.dialogue_turns)),
            temperature=args.temperature,
            max_input_chars=max(5000, int(args.max_input_chars)),
            chunk_size=max(2000, int(args.chunk_size)),
            logger=logger,
        )

        if args.skip_tts:
            logger.info("Skipping TTS by user request (--skip-tts).")
            logger.info("Completed. Dialogue file: %s", dialogue_path)
            return 0

        audio_path = synthesize_audio_from_dialogue(
            dialogue_path=dialogue_path,
            api_keys=voice_api_keys,
            tts_model=args.tts_model,
            host_a_voice=args.host_a_voice,
            host_b_voice=args.host_b_voice,
            output_dir=paper_dir,
            logger=logger,
        )
        logger.info("Completed. Final audio: %s", audio_path)
        return 0

    except (MainError, LLMWriterError, TTSAudioError, FileNotFoundError, ValueError) as exc:
        logging.getLogger("paper_suite").exception("Pipeline failed: %s", exc)
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("paper_suite").exception("Unexpected failure: %s", exc)
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
