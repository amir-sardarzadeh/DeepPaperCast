#!/usr/bin/env python3
"""Standalone LaTeX explainer generator (no dependency on llm_writer.py)."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Quick-start config (edit then run: python latex.py)
DETAIL_LEVEL = "High"
Company = "Claude"
Model = "claude-sonnet-4-6"
API_FILE = "apit.txt"
FINAL_ROOT = "Final"
File = "3.pdf"

SUPPORTED_MODELS = {
    "claude-3-7-sonnet-latest": {"provider": "anthropic", "max_context_tokens": 200_000, "max_output_tokens": 128_000, "supports_extended_thinking": True},
    "claude-opus-4-6": {"provider": "anthropic", "max_context_tokens": 200_000, "max_output_tokens": 128_000, "supports_extended_thinking": True},
    "claude-sonnet-4-6": {"provider": "anthropic", "max_context_tokens": 200_000, "max_output_tokens": 128_000, "supports_extended_thinking": True},
    "gpt-4o": {"provider": "openai", "max_context_tokens": 128_000, "max_output_tokens": 16384, "supports_extended_thinking": False},
    "gpt-4o-latest": {"provider": "openai", "max_context_tokens": 128_000, "max_output_tokens": 16384, "supports_extended_thinking": False},
}

LATEX_SYSTEM_PROMPT = r"""You are an expert technical educator.
Write a highly comprehensive explanation of the provided academic paper for a third-year undergraduate engineering or science student.

Requirements:
- Teach background concepts before introducing the paper's novelty.
- Explain methodology step-by-step with intuition and rigor.
- Explain key equations clearly, defining symbols and interpreting the math.
- Include limitations and practical implications.
- Output ONLY valid standalone LaTeX code, including:
  - \documentclass{article}
  - required packages (at least amsmath, amssymb, geometry, hyperref)
  - \begin{document} ... \end{document}
- Do not include markdown fences or any non-LaTeX text.
"""

INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
ANTHROPIC_STREAMING_MAX_TOKENS_THRESHOLD = 21333


class LatexError(RuntimeError):
    """Raised for LaTeX pipeline failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX explanation PDF from a paper PDF.")
    parser.add_argument("--pdf", default=File, help="Input PDF path")
    parser.add_argument("--detail-level", choices=["Default", "High"], default=DETAIL_LEVEL)
    parser.add_argument("--company", default=Company)
    parser.add_argument("--model", default=Model)
    parser.add_argument("--api-file", default=API_FILE)
    parser.add_argument("--final-root", default=FINAL_ROOT)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("latex")
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
        raise LatexError(f"API key file not found: {api_file}")

    keys: Dict[str, str] = {}
    for line_no, raw_line in enumerate(api_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LatexError(f"Malformed line {line_no} in {api_file.name}: {line}")
        k, v = line.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if not k or not v:
            raise LatexError(f"Malformed line {line_no} in {api_file.name}: {line}")
        keys[k] = v
    return keys


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise LatexError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def sanitize_name(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def extract_pdf_text(pdf_path: Path, logger: logging.Logger) -> tuple[Optional[str], str]:
    if not pdf_path.exists():
        raise LatexError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise LatexError(f"Input is not a PDF: {pdf_path}")

    title: Optional[str] = None
    text = ""

    try:
        from pypdf import PdfReader

        logger.info("Extracting text with pypdf: %s", pdf_path)
        reader = PdfReader(str(pdf_path))
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip() or None
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf extraction failed: %s", exc)

    if len(text) < 500:
        try:
            import pdfplumber

            logger.info("Falling back to pdfplumber extraction.")
            with pdfplumber.open(str(pdf_path)) as pdf:
                alt = "\n".join((p.extract_text() or "") for p in pdf.pages).strip()
            if len(alt) > len(text):
                text = alt
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber extraction failed: %s", exc)

    if not text:
        raise LatexError("Failed to extract readable text from PDF.")
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
    raise LatexError("Company must be OpenAI or Claude.")


def _lookup_model_profile(model_name: str, provider: str) -> Dict[str, object]:
    for key, value in SUPPORTED_MODELS.items():
        if key.lower() == model_name.lower():
            return value
    return {
        "provider": provider,
        "max_context_tokens": 200_000 if provider == "anthropic" else 128_000,
        "max_output_tokens": 128_000 if provider == "anthropic" else 16_384,
        "supports_extended_thinking": provider == "anthropic",
    }


def calculate_dynamic_budgets(input_tokens: int, context_window: int, detail_level: str, supports_thinking: bool) -> tuple[int, int]:
    safety_margin = max(1500, int(context_window * 0.03))
    available = max(1024, context_window - input_tokens - safety_margin)
    if detail_level.lower() == "high":
        max_output = available
        thinking = available if supports_thinking else 0
    else:
        max_output = max(1024, min(8192, available))
        thinking = min(max_output // 2, 8000) if supports_thinking else 0
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
    def __init__(self, *, model: str, api_key: str, base_url: Optional[str]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LatexError("Missing dependency 'openai'. Install requirements.txt.") from exc
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

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
                raise LatexError(f"OpenAI API error: {exc}") from exc

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
        raise LatexError("OpenAI response did not contain readable text.")


class AnthropicMessagesClient(BaseLLMClient):
    def __init__(self, *, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LatexError("Missing dependency 'anthropic'. Install requirements.txt.") from exc
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

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
                    raise LatexError(f"Anthropic API error: {retry_exc}") from retry_exc
            else:
                raise LatexError(f"Anthropic API error: {exc}") from exc

        parts: List[str] = []
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                parts.append(block.text)
        if parts:
            return "\n".join(parts).strip()
        raise LatexError("Anthropic response did not contain readable text.")


def build_llm_client(
    *,
    provider: str,
    model: str,
    api_keys: Dict[str, str],
    llm_base_url: Optional[str],
) -> BaseLLMClient:
    p = provider.lower()
    if p == "openai":
        return OpenAIResponsesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENAI_API_KEY", "OPENAI"]),
            base_url=llm_base_url,
        )
    if p == "anthropic":
        return AnthropicMessagesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["ANTHROPIC_API_KEY", "ANTHROPIC"]),
        )
    raise LatexError(f"Unsupported provider: {provider}")


def compile_latex(tex_path: Path, logger: logging.Logger) -> Path:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise LatexError("pdflatex not found in PATH. Install MiKTeX, TeX Live, or MacTeX.")

    cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tex_path.parent),
        str(tex_path),
    ]
    logger.info("Running pdflatex...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("pdflatex stdout:\n%s", result.stdout.strip())
        logger.error("pdflatex stderr:\n%s", result.stderr.strip())
        raise LatexError("LaTeX compilation failed. Check processing.log.")

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise LatexError("pdflatex finished but no PDF was produced.")
    return pdf_path


def run() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    try:
        pdf_path = Path(args.pdf).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = (root_dir / pdf_path).resolve()

        provider = _resolve_provider(args.company)
        model = args.model

        boot_logger = logging.getLogger("latex_boot")
        boot_logger.setLevel(logging.INFO)
        boot_logger.handlers.clear()
        boot_logger.addHandler(logging.StreamHandler(sys.stdout))

        title, text = extract_pdf_text(pdf_path, boot_logger)
        # For this utility, always use the source PDF filename as the folder/file base.
        paper_name = sanitize_name(pdf_path.stem)

        final_root = Path(args.final_root).expanduser()
        if not final_root.is_absolute():
            final_root = (root_dir / final_root).resolve()
        paper_dir = final_root / paper_name
        paper_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(paper_dir / "processing.log")
        logger.info("LaTeX run started for paper: %s", paper_name)

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
            supports_thinking=bool(profile["supports_extended_thinking"]),
        )
        if args.max_output_tokens is not None:
            max_output = int(args.max_output_tokens)
        if args.thinking_budget is not None:
            thinking = int(args.thinking_budget)
        model_output_cap = int(profile["max_output_tokens"])
        if max_output > model_output_cap:
            logger.warning(
                "Clamping max_output_tokens from %d to model cap %d for %s.",
                max_output,
                model_output_cap,
                model,
            )
        max_output = max(256, min(max_output, model_output_cap))
        if thinking > max_output:
            logger.warning(
                "Clamping thinking_budget from %d to %d to fit max_output_tokens.",
                thinking,
                max_output,
            )
        thinking = max(0, min(thinking, max_output))
        if provider != "anthropic":
            thinking = 0

        user_prompt = (
            f"Paper title: {paper_name}\n\n"
            "Write a complete LaTeX teaching document for this paper.\n\n"
            "Paper content:\n"
            f"{text}"
        )

        client = build_llm_client(
            provider=provider,
            model=model,
            api_keys=api_keys,
            llm_base_url=args.llm_base_url,
        )
        latex_code = client.generate_text(
            system_prompt=LATEX_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=float(args.temperature),
            max_output_tokens=max(256, int(max_output)),
            thinking_budget_tokens=max(0, int(thinking)),
            enable_extended_thinking=(provider == "anthropic" and thinking > 0),
        )

        tex_path = paper_dir / f"{paper_name}_explained.tex"
        tex_path.write_text(latex_code.strip() + "\n", encoding="utf-8")
        logger.info("Saved tex file: %s", tex_path)

        pdf_out = compile_latex(tex_path, logger)
        logger.info("Compiled PDF: %s", pdf_out)
        return 0

    except (LatexError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
