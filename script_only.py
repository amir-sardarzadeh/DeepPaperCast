#!/usr/bin/env python3
"""Standalone transcript generator (no TTS, no dependency on llm_writer.py)."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Quick-start config (edit then run: python script_only.py)
DETAIL_LEVEL = "High"
Company = "Claude"
Model = "claude-opus-4-6"
API_FILE = "api.txt"
FINAL_ROOT = "Final"
File = "Channel_Parameter_Estimation_and_Localization_for_Near-field_XL-MIMO_Communications.pdf"

SUPPORTED_MODELS = {
    "claude-3-7-sonnet-latest": {"provider": "anthropic", "max_context_tokens": 200_000, "supports_extended_thinking": True},
    "claude-opus-4-6": {"provider": "anthropic", "max_context_tokens": 200_000, "supports_extended_thinking": True},
    "claude-sonnet-4-6": {"provider": "anthropic", "max_context_tokens": 200_000, "supports_extended_thinking": True},
    "gpt-4o": {"provider": "openai", "max_context_tokens": 128_000, "supports_extended_thinking": False},
    "gpt-4o-latest": {"provider": "openai", "max_context_tokens": 128_000, "supports_extended_thinking": False},
}

DEFAULT_DIALOGUE_SYSTEM_PROMPT = """You are a world-class podcast producer and scriptwriter.
Write a lively, engaging two-person podcast transcript between Host A and Host B about the provided academic paper.

Requirements:
- Keep the conversation highly engaging, conversational, and accessible.
- Host A is the structured guide; Host B is the insightful color commentator.
- Cover: problem, methodology, key findings, limitations, and practical implications.
- Formula Handling: If there are simple, foundational formulas, explain them intuitively using physical or visual analogies. Skip over highly complex, impenetrable math and just focus on the resulting concepts.
- Keep each speaking turn short and punchy (1-3 sentences maximum).
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions, or narrator notes. Spell out symbols.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

HIGH_DETAIL_DIALOGUE_SYSTEM_PROMPT = """You are a world-class podcast producer and scriptwriter.
Write a highly comprehensive, extended-length, two-person podcast transcript between Host A and Host B about the provided academic paper.

Requirements:
- Maximize your output. Take the time to explain the background context and foundational concepts before diving into the paper's novel contributions.
- Detail the methodology step-by-step. Discuss why the authors chose this approach and alternatives.
- Host A acts as the expert guide. Host B acts as the highly curious learner asking probing follow-up questions.
- STRICT Formula Handling: You MUST explain all formulas presented in the paper. When discussing a formula, you must explicitly state its number as it appears in the text (for example: "If we look at Equation 4..."). Translate these complex equations into physical, visual, or relatable analogies so they make sense in an audio format. DO NOT just read the raw math symbols.
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions, or narrator notes. Spell out symbols.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

CHUNK_SUMMARY_SYSTEM_PROMPT = """You are an expert research assistant preparing notes for a long-form technical explanation.
Summarize the provided section of an academic paper.

Return concise plain text with these headings:
1) Core Claim
2) Methods
3) Key Evidence
4) Caveats
5) Notation and Equations Mentioned
"""

SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)
INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
ANTHROPIC_STREAMING_MAX_TOKENS_THRESHOLD = 21333


class ScriptOnlyError(RuntimeError):
    """Raised for script_only processing failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate podcast transcript only.")
    parser.add_argument("--pdf", default=File, help="Input PDF path")
    parser.add_argument("--detail-level", choices=["Default", "High"], default=DETAIL_LEVEL)
    parser.add_argument("--company", default=Company)
    parser.add_argument("--model", default=Model)
    parser.add_argument("--api-file", default=API_FILE)
    parser.add_argument("--final-root", default=FINAL_ROOT)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dialogue-turns", type=int, default=48)
    parser.add_argument("--max-input-chars", type=int, default=300000)
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("script_only")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def load_api_keys(api_file: Path) -> Dict[str, str]:
    if not api_file.exists():
        raise ScriptOnlyError(f"API key file not found: {api_file}")

    keys: Dict[str, str] = {}
    for line_no, raw_line in enumerate(api_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ScriptOnlyError(f"Malformed line {line_no} in {api_file.name}: {line}")
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            raise ScriptOnlyError(f"Malformed line {line_no} in {api_file.name}: {line}")
        keys[key] = value
    return keys


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise ScriptOnlyError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def sanitize_name(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def extract_pdf_text(pdf_path: Path, logger: logging.Logger) -> tuple[Optional[str], str]:
    if not pdf_path.exists():
        raise ScriptOnlyError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ScriptOnlyError(f"Input is not a PDF: {pdf_path}")

    title: Optional[str] = None
    text = ""
    try:
        from pypdf import PdfReader

        logger.info("Extracting text with pypdf: %s", pdf_path)
        reader = PdfReader(str(pdf_path))
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip() or None
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf extraction failed: %s", exc)

    if len(text) < 500:
        try:
            import pdfplumber

            logger.info("Falling back to pdfplumber extraction.")
            with pdfplumber.open(str(pdf_path)) as pdf:
                alt = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
            if len(alt) > len(text):
                text = alt
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber extraction failed: %s", exc)

    if not text:
        raise ScriptOnlyError("Failed to extract readable text from PDF.")
    return title, text


def choose_paper_name(pdf_path: Path, extracted_title: Optional[str], text: str) -> str:
    if extracted_title:
        lowered = extracted_title.lower().strip()
        if lowered not in {"untitled", "document", "microsoft word -"}:
            return sanitize_name(extracted_title)

    for line in text.splitlines()[:40]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if 20 <= len(candidate) <= 180 and not candidate.lower().startswith("arxiv"):
            return sanitize_name(candidate)
    return sanitize_name(pdf_path.stem)


def estimate_tokens_from_text(text: str, model_name: str) -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:  # noqa: BLE001
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def _resolve_provider(company: str) -> str:
    value = (company or "").strip().lower()
    if "claude" in value or "anthropic" in value:
        return "anthropic"
    if "openai" in value or "gpt" in value or "chatgpt" in value:
        return "openai"
    raise ScriptOnlyError("Company must be OpenAI or Claude.")


def _lookup_model_profile(model_name: str, provider: str) -> Dict[str, object]:
    for key, value in SUPPORTED_MODELS.items():
        if key.lower() == model_name.lower():
            return value
    return {
        "provider": provider,
        "max_context_tokens": 200_000 if provider == "anthropic" else 128_000,
        "supports_extended_thinking": provider == "anthropic",
    }


def calculate_dynamic_budgets(
    *,
    input_tokens: int,
    context_window: int,
    detail_level: str,
    supports_extended_thinking: bool,
) -> Tuple[int, int]:
    safety_margin = max(1500, int(context_window * 0.03))
    available = max(1024, context_window - input_tokens - safety_margin)
    if detail_level.lower() == "high":
        max_output = available
        thinking = available if supports_extended_thinking else 0
    else:
        max_output = max(1024, min(8192, available))
        thinking = min(max_output // 2, 8000) if supports_extended_thinking else 0
    return max_output, thinking


class BaseLLMClient:
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_budget_tokens: int,
        enable_extended_thinking: bool,
    ) -> str:
        raise NotImplementedError


class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, *, model: str, api_key: str, base_url: Optional[str], logger: logging.Logger) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ScriptOnlyError("Missing dependency 'openai'. Install requirements.txt.") from exc

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.logger = logger

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_budget_tokens: int,
        enable_extended_thinking: bool,
    ) -> str:
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        try:
            resp = self.client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in str(exc).lower():
                payload.pop("temperature", None)
                resp = self.client.responses.create(**payload)
            else:
                raise ScriptOnlyError(f"OpenAI API error: {exc}") from exc

        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        out = getattr(resp, "output", None) or []
        parts: List[str] = []
        for item in out:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    ctype = getattr(content, "type", None)
                    if ctype in {"output_text", "text"} and getattr(content, "text", None):
                        parts.append(content.text)
        if parts:
            return "\n".join(parts).strip()
        raise ScriptOnlyError("OpenAI response did not include readable text.")


class AnthropicMessagesClient(BaseLLMClient):
    def __init__(self, *, model: str, api_key: str, logger: logging.Logger) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ScriptOnlyError("Missing dependency 'anthropic'. Install requirements.txt.") from exc
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.logger = logger

    def _send(self, payload: Dict[str, object]) -> object:
        max_tokens = int(payload.get("max_tokens", 0) or 0)
        if max_tokens >= ANTHROPIC_STREAMING_MAX_TOKENS_THRESHOLD:
            with self.client.messages.stream(**payload) as stream:
                return stream.get_final_message()
        return self.client.messages.create(**payload)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_budget_tokens: int,
        enable_extended_thinking: bool,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
        }
        if enable_extended_thinking and thinking_budget_tokens > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget_tokens)}
        else:
            payload["temperature"] = temperature

        try:
            message = self._send(payload)
        except Exception as exc:  # noqa: BLE001
            if "thinking" in payload:
                payload.pop("thinking", None)
                payload["temperature"] = temperature
                try:
                    message = self._send(payload)
                except Exception as retry_exc:  # noqa: BLE001
                    raise ScriptOnlyError(f"Anthropic API error: {retry_exc}") from retry_exc
            else:
                raise ScriptOnlyError(f"Anthropic API error: {exc}") from exc

        parts: List[str] = []
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                parts.append(block.text)
        if parts:
            return "\n".join(parts).strip()
        raise ScriptOnlyError("Anthropic response did not include readable text.")


def build_llm_client(
    *,
    provider: str,
    model: str,
    api_keys: Dict[str, str],
    llm_base_url: Optional[str],
    logger: logging.Logger,
) -> BaseLLMClient:
    p = provider.lower()
    if p == "openai":
        return OpenAIResponsesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENAI_API_KEY", "OPENAI"]),
            base_url=llm_base_url,
            logger=logger,
        )
    if p == "anthropic":
        return AnthropicMessagesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["ANTHROPIC_API_KEY", "ANTHROPIC"]),
            logger=logger,
        )
    raise ScriptOnlyError(f"Unsupported provider: {provider}")


def _split_long_block(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                out.append(current)
                current = ""
            words = sentence.split()
            bucket: List[str] = []
            size = 0
            for word in words:
                add = len(word) + (1 if bucket else 0)
                if size + add > max_chars and bucket:
                    out.append(" ".join(bucket))
                    bucket = [word]
                    size = len(word)
                else:
                    bucket.append(word)
                    size += add
            if bucket:
                out.append(" ".join(bucket))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def maybe_summarize_long_text(
    *,
    llm_client: BaseLLMClient,
    text: str,
    max_input_chars: int,
    chunk_size: int,
) -> str:
    if len(text) <= max_input_chars:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for para in paragraphs:
        if len(para) > chunk_size:
            if cur:
                chunks.append("\n\n".join(cur))
                cur = []
                cur_len = 0
            chunks.extend(_split_long_block(para, chunk_size))
            continue
        projected = cur_len + len(para) + 2
        if projected > chunk_size and cur:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(para)
        cur_len += len(para) + 2
    if cur:
        chunks.append("\n\n".join(cur))

    summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = llm_client.generate_text(
            system_prompt=CHUNK_SUMMARY_SYSTEM_PROMPT,
            user_prompt=f"Chunk {idx}/{len(chunks)}:\n\n{chunk}",
            temperature=0.2,
            max_output_tokens=1200,
            thinking_budget_tokens=0,
            enable_extended_thinking=False,
        )
        summaries.append(f"Chunk {idx}:\n{summary}")
    return "\n\n".join(summaries)


def normalize_dialogue(raw: str) -> List[str]:
    lines: List[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if match:
            speaker = "Host A" if "a" in match.group(1).lower() else "Host B"
            content = match.group(2).strip()
            if content:
                lines.append(f"{speaker}: {content}")
        elif lines:
            lines[-1] = f"{lines[-1]} {line}".strip()
    if not lines:
        raise ScriptOnlyError("LLM did not return parseable Host A / Host B lines.")
    return lines


def run() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    try:
        pdf_path = Path(args.pdf).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = (root_dir / pdf_path).resolve()

        provider = _resolve_provider(args.company)
        model = args.model

        temp_logger = logging.getLogger("script_only_boot")
        temp_logger.setLevel(logging.INFO)
        temp_logger.handlers.clear()
        temp_logger.addHandler(logging.StreamHandler(sys.stdout))

        title, text = extract_pdf_text(pdf_path, temp_logger)
        # Match latex.py behavior: always use source PDF filename as output folder/file base.
        paper_name = sanitize_name(pdf_path.stem)

        final_root = Path(args.final_root).expanduser()
        if not final_root.is_absolute():
            final_root = (root_dir / final_root).resolve()
        paper_dir = final_root / paper_name
        paper_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(paper_dir / "processing.log")
        logger.info("Script-only run started for paper: %s", paper_name)

        copied_pdf = paper_dir / pdf_path.name
        if copied_pdf.resolve() != pdf_path.resolve():
            shutil.copy2(pdf_path, copied_pdf)

        api_keys = load_api_keys(Path(args.api_file).expanduser().resolve())
        profile = _lookup_model_profile(model, provider)
        input_tokens = estimate_tokens_from_text(text, model)
        max_output, thinking = calculate_dynamic_budgets(
            input_tokens=input_tokens,
            context_window=int(profile["max_context_tokens"]),
            detail_level=args.detail_level,
            supports_extended_thinking=bool(profile["supports_extended_thinking"]),
        )

        if args.max_output_tokens is not None:
            max_output = int(args.max_output_tokens)
        if args.thinking_budget is not None:
            thinking = int(args.thinking_budget)
        if provider != "anthropic":
            thinking = 0

        client = build_llm_client(
            provider=provider,
            model=model,
            api_keys=api_keys,
            llm_base_url=args.llm_base_url,
            logger=logger,
        )

        llm_input = maybe_summarize_long_text(
            llm_client=client,
            text=text,
            max_input_chars=max(5000, int(args.max_input_chars)),
            chunk_size=max(2000, int(args.chunk_size)),
        )

        system_prompt = HIGH_DETAIL_DIALOGUE_SYSTEM_PROMPT if args.detail_level.lower() == "high" else DEFAULT_DIALOGUE_SYSTEM_PROMPT
        user_prompt = (
            f"Paper title: {paper_name}\n\n"
            f"Generate approximately {max(12, int(args.dialogue_turns))} turns total.\n"
            "Allowed output format only:\n"
            "Host A: <text>\n"
            "Host B: <text>\n\n"
            "Paper content:\n"
            f"{llm_input}"
        )

        raw = client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(args.temperature),
            max_output_tokens=max(256, int(max_output)),
            thinking_budget_tokens=max(0, int(thinking)),
            enable_extended_thinking=(provider == "anthropic" and thinking > 0),
        )
        lines = normalize_dialogue(raw)

        dialogue_path = paper_dir / f"{paper_name}.txt"
        dialogue_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        logger.info("Dialogue saved: %s", dialogue_path)
        return 0

    except (ScriptOnlyError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
