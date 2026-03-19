#!/usr/bin/env python3
"""Orchestrator for converting an academic paper PDF into a 2-host AI podcast."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from llm_writer import DialogueResult, generate_dialogue_file_from_pdf
from tts_audio import synthesize_audio_from_dialogue

# ---------------------------------------------------------------------------
# Quick-start settings (edit these 3 values, then run: python main.py)
# ---------------------------------------------------------------------------
# Example:
# File = "my_paper.pdf"
# Company = "Claude"   # or "OpenAI"
# Model = "claude-sonnet-4-5" "claude-sonnet-4-6" "gpt-5.4-pro"
# ExtendedThinking = True  # Anthropic only
# ThinkingBudgetTokens = 12000  # Anthropic only
# DETAIL_LEVEL = "Default" or "High"
File = "2401.02844v1.pdf"
Company = "Claude"
Model = "claude-sonnet-4-6"
ExtendedThinking = True
ThinkingBudgetTokens = 12000
DETAIL_LEVEL = "High"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the podcast generation workflow."""
    parser = argparse.ArgumentParser(
        description="Generate a two-person AI podcast dialogue and audio from an academic PDF."
    )
    parser.add_argument("--pdf", "--file", dest="pdf", default=None, help="Path to the input PDF.")
    parser.add_argument(
        "--api-writer-file",
        default=None,
        help="Path to writer API key file (default: script_folder/apit.txt).",
    )
    parser.add_argument(
        "--api-voice-file",
        default=None,
        help="Path to voice API key file (default: script_folder/apiv.txt).",
    )
    parser.add_argument("--log-dir", default="logs", help="Directory where run logs are stored.")

    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["openai", "anthropic", "openrouter", "openai_compatible"],
        help="Provider used for dialogue generation.",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Shortcut for provider: 'OpenAI' -> openai, 'Claude' -> anthropic.",
    )
    parser.add_argument("--llm-model", "--model", dest="llm_model", default=None, help="LLM model name.")
    parser.add_argument(
        "--detail-level",
        default=None,
        choices=["Default", "High"],
        help="Script detail level for dialogue generation.",
    )
    parser.add_argument(
        "--extended-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Anthropic extended thinking (Anthropic models only).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Anthropic extended thinking budget tokens (Anthropic only).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional base URL for openai_compatible or custom OpenAI-compatible endpoints.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Dialogue generation temperature.")
    parser.add_argument("--dialogue-turns", type=int, default=48, help="Approximate number of dialogue turns.")
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=300000,
        help="If extracted text exceeds this size, chunk-and-summarize before final dialogue.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12000,
        help="Chunk size used during summarize-before-dialogue generation.",
    )

    parser.add_argument(
        "--tts-provider",
        default="openai",
        choices=["openai", "elevenlabs"],
        help="Provider used for text-to-speech.",
    )
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts", help="TTS model id.")
    parser.add_argument("--host-a-voice", default="echo", help="Voice for Person/Host A (male).")
    parser.add_argument("--host-b-voice", default="sage", help="Voice for Person/Host B (female).")
    parser.add_argument(
        "--output-format",
        default="mp3",
        choices=["mp3"],
        help="Final audio format (fixed to mp3 at 48 kHz / 320 kbps).",
    )

    return parser.parse_args()


def _resolve_provider(llm_provider: str | None, company: str | None) -> str:
    """Resolve final LLM provider from explicit provider or Company shortcut."""
    if llm_provider:
        return llm_provider

    company_value = (company or "").strip().lower()
    if not company_value:
        raise ValueError("Set Company at the top of main.py (OpenAI or Claude), or pass --llm-provider.")
    if "claude" in company_value or "anthropic" in company_value:
        return "anthropic"
    if "openai" in company_value or "chatgpt" in company_value:
        return "openai"
    raise ValueError(f"Unsupported Company value '{company}'. Use OpenAI or Claude.")


def _resolve_pdf_path(pdf_value: str | None, script_dir: Path) -> Path:
    """Resolve PDF path from CLI or File config, allowing relative names in script folder."""
    if not pdf_value:
        raise ValueError("Set File at the top of main.py or pass --pdf.")

    candidate = Path(pdf_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    direct = candidate.resolve()
    if direct.exists():
        return direct

    script_candidate = (script_dir / candidate).resolve()
    if script_candidate.exists():
        return script_candidate

    # Fallback: file may have been moved into an output folder after a previous run.
    matches = sorted(script_dir.rglob(candidate.name))
    if matches:
        return matches[0].resolve()
    return script_candidate


def _resolve_model(model_value: str | None) -> str:
    """Resolve LLM model from CLI or Model config."""
    model = (model_value or "").strip()
    if not model:
        raise ValueError("Set Model at the top of main.py or pass --model/--llm-model.")
    return model


def _resolve_detail_level(detail_level_value: str | None) -> str:
    """Resolve and validate detail level value."""
    value = (detail_level_value or "").strip().lower()
    if not value:
        raise ValueError("Set DETAIL_LEVEL at the top of main.py or pass --detail-level.")
    if value == "default":
        return "Default"
    if value == "high":
        return "High"
    raise ValueError("DETAIL_LEVEL must be either 'Default' or 'High'.")


def _normalize_model_for_provider(provider: str, model: str) -> str:
    """Allow simple model shortcuts like 'Sonnet' and normalize to provider model ids."""
    p = provider.lower()
    m = model.strip().lower()

    if p == "anthropic":
        if m in {"sonnet", "claude sonnet", "claude-sonnet"}:
            return "claude-sonnet-4-5"
        if m in {"opus", "claude opus", "claude-opus"}:
            return "claude-opus-4-1"

    return model


def setup_logger(log_dir: Path) -> Tuple[logging.Logger, Path]:
    """Create a console+file logger for the current run and return logger + log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"podcast_generator_{timestamp}.log"

    logger = logging.getLogger("podcast_generator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def read_api_keys(api_file: Path) -> Dict[str, str]:
    """Read API keys from a text file with SERVICE_NAME=api_key lines."""
    if not api_file.exists():
        raise FileNotFoundError(f"API key file not found: {api_file}")

    keys: Dict[str, str] = {}
    with api_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Malformed {api_file.name} line {line_no}: {raw_line.strip()}")
            service, value = line.split("=", 1)
            service = service.strip().upper()
            value = value.strip()
            if not service:
                raise ValueError(f"Malformed {api_file.name} line {line_no}: missing service name.")
            if not value:
                raise ValueError(f"Malformed {api_file.name} line {line_no}: missing API key value.")
            keys[service] = value
    return keys


def organize_outputs(
    *,
    paper_name: str,
    pdf_path: Path,
    dialogue_path: Path,
    audio_path: Path,
    final_root: Path,
    logger: logging.Logger,
) -> Path:
    """Move the PDF, dialogue text, and audio file into a folder named after the paper."""
    final_root.mkdir(parents=True, exist_ok=True)
    target_dir = final_root / paper_name
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Organizing outputs into folder: %s", target_dir)

    for source in [pdf_path, dialogue_path, audio_path]:
        target = target_dir / source.name
        if source.resolve() == target.resolve():
            logger.debug("File already in target folder: %s", source)
            continue
        if target.exists():
            logger.warning("Overwriting existing file in target folder: %s", target)
            target.unlink()
        shutil.move(str(source), str(target))
        logger.info("Moved %s -> %s", source.name, target)

    return target_dir


def run() -> int:
    """Execute the end-to-end pipeline and return process exit code."""
    args = parse_args()
    logger, log_path = setup_logger(Path(args.log_dir))
    logger.info("Run started. Log file: %s", log_path.resolve())

    try:
        script_dir = Path(__file__).resolve().parent
        resolved_pdf_value = args.pdf if args.pdf else File
        resolved_company = args.company if args.company else Company
        resolved_model = _resolve_model(args.llm_model if args.llm_model else Model)
        resolved_detail_level = _resolve_detail_level(args.detail_level if args.detail_level else DETAIL_LEVEL)
        resolved_provider = _resolve_provider(args.llm_provider, resolved_company)
        resolved_model = _normalize_model_for_provider(resolved_provider, resolved_model)
        resolved_extended_thinking = (
            args.extended_thinking if args.extended_thinking is not None else ExtendedThinking
        )
        resolved_thinking_budget = (
            args.thinking_budget if args.thinking_budget is not None else ThinkingBudgetTokens
        )
        pdf_path = _resolve_pdf_path(resolved_pdf_value, script_dir)
        writer_api_file = (
            Path(args.api_writer_file).expanduser().resolve()
            if args.api_writer_file
            else (script_dir / "apit.txt").resolve()
        )
        voice_api_file = (
            Path(args.api_voice_file).expanduser().resolve()
            if args.api_voice_file
            else (script_dir / "apiv.txt").resolve()
        )

        logger.info(
            "Resolved config: file=%s, company=%s, provider=%s, model=%s, detail_level=%s, extended_thinking=%s, thinking_budget=%s",
            pdf_path,
            resolved_company,
            resolved_provider,
            resolved_model,
            resolved_detail_level,
            resolved_extended_thinking,
            resolved_thinking_budget,
        )

        logger.info("Loading writer API keys from: %s", writer_api_file)
        writer_api_keys = read_api_keys(writer_api_file)
        logger.info("Loaded writer API services: %s", ", ".join(sorted(writer_api_keys.keys())))

        logger.info("Loading voice API keys from: %s", voice_api_file)
        voice_api_keys = read_api_keys(voice_api_file)
        logger.info("Loaded voice API services: %s", ", ".join(sorted(voice_api_keys.keys())))

        logger.info("Generating podcast dialogue from PDF.")
        dialogue_result: DialogueResult = generate_dialogue_file_from_pdf(
            pdf_path=pdf_path,
            api_keys=writer_api_keys,
            llm_provider=resolved_provider,
            llm_model=resolved_model,
            llm_base_url=args.llm_base_url,
            temperature=args.temperature,
            dialogue_turns=max(36, args.dialogue_turns),
            max_input_chars=max(5000, args.max_input_chars),
            chunk_size=max(2000, args.chunk_size),
            detail_level=resolved_detail_level,
            anthropic_extended_thinking=bool(resolved_extended_thinking),
            anthropic_thinking_budget_tokens=max(1024, int(resolved_thinking_budget)),
            output_dir=pdf_path.parent,
            logger=logger,
        )
        logger.info("Dialogue file created: %s", dialogue_result.dialogue_path)

        logger.info("Synthesizing audio from generated dialogue.")
        audio_path = synthesize_audio_from_dialogue(
            dialogue_path=dialogue_result.dialogue_path,
            api_keys=voice_api_keys,
            tts_provider=args.tts_provider,
            tts_model=args.tts_model,
            host_a_voice=args.host_a_voice,
            host_b_voice=args.host_b_voice,
            output_format=args.output_format,
            output_dir=pdf_path.parent,
            logger=logger,
        )
        logger.info("Audio file created: %s", audio_path)

        output_folder = organize_outputs(
            paper_name=dialogue_result.paper_name,
            pdf_path=pdf_path,
            dialogue_path=dialogue_result.dialogue_path,
            audio_path=audio_path,
            final_root=script_dir / "Final",
            logger=logger,
        )

        logger.info("Pipeline completed successfully.")
        logger.info("Final output folder: %s", output_folder)
        logger.info("Run log file: %s", log_path.resolve())
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        logger.error("See log file for details: %s", log_path.resolve())
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
