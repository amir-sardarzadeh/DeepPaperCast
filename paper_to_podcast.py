#!/usr/bin/env python3
"""
Build a two-host AI podcast from an academic paper PDF.

Usage example:
  python paper_to_podcast.py ^
    --pdf "C:\\path\\to\\paper.pdf" ^
    --llm-provider openai ^
    --llm-model gpt-5.4-pro ^
    --tts-provider openai ^
    --tts-model gpt-4o-mini-tts ^
    --host-a-voice echo ^
    --host-b-voice sage ^
    --output-format mp3

Environment variables (.env supported):
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  OPENROUTER_API_KEY
  OPENAI_COMPAT_API_KEY
  ELEVENLABS_API_KEY

Notes:
  - Requires ffmpeg installed on your system for pydub export.
  - File/folder names are sanitized to be filesystem-safe on Windows/macOS/Linux.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DIALOGUE_SYSTEM_PROMPT = """You are a world-class podcast writer.
Write a lively two-person conversation between Host A and Host B about an academic paper.

Requirements:
- The conversation must be engaging and natural, like an expert "deep dive" podcast.
- Explain technical ideas clearly for non-experts without losing rigor.
- Cover: problem, method, key findings, limitations, and practical implications.
- Include occasional clarifying questions, examples, and mild disagreement.
- Keep each turn concise (1-3 sentences).
- DO NOT include stage directions, markdown, or narrator text.
- STRICT FORMAT: each line must begin with exactly "Host A:" or "Host B:".
- Alternate speakers naturally; start with Host A.
"""

CHUNK_SUMMARY_SYSTEM_PROMPT = """You summarize sections of academic papers for downstream dialogue generation.
Return concise plain text with these headings:
1) Core claim
2) Methods
3) Key evidence/results
4) Caveats/limits
5) Real-world significance
"""

SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class DialogueTurn:
    speaker: str  # Host A or Host B
    text: str


@dataclass
class Args:
    pdf: Path
    llm_provider: str
    llm_model: str
    llm_base_url: Optional[str]
    tts_provider: str
    tts_model: str
    host_a_voice: str
    host_b_voice: str
    output_format: str
    dialogue_turns: int
    temperature: float
    max_input_chars: int
    chunk_size: int


class PipelineError(RuntimeError):
    pass


class BaseLLMClient:
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        raise NotImplementedError


class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("Missing dependency 'openai'. Install dependencies from requirements.txt.") from exc

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
    ) -> str:
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        try:
            response = self.client.responses.create(**payload)
        except Exception as exc:
            # Some models reject temperature; retry without it.
            if "temperature" in str(exc).lower():
                payload.pop("temperature", None)
                response = self.client.responses.create(**payload)
            else:
                raise PipelineError(f"OpenAI Responses API failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        recovered = _recover_openai_response_text(response)
        if recovered:
            return recovered.strip()
        raise PipelineError("OpenAI response did not contain readable text.")


class OpenAIChatCompatibleClient(BaseLLMClient):
    """For OpenRouter and generic OpenAI-compatible chat endpoints."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("Missing dependency 'openai'. Install dependencies from requirements.txt.") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        try:
            completion = self.client.chat.completions.create(**payload)
        except Exception as exc:
            if "temperature" in str(exc).lower():
                payload.pop("temperature", None)
                completion = self.client.chat.completions.create(**payload)
            else:
                raise PipelineError(f"OpenAI-compatible chat API failed: {exc}") from exc

        content = completion.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()
        raise PipelineError("Chat completion did not contain readable text.")


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise PipelineError("Missing dependency 'anthropic'. Install dependencies from requirements.txt.") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise PipelineError(f"Anthropic Messages API failed: {exc}") from exc

        text_parts: List[str] = []
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
        if text_parts:
            return "\n".join(text_parts).strip()
        raise PipelineError("Anthropic response did not contain readable text.")


class BaseTTSClient:
    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        raise NotImplementedError


class OpenAITTSClient(BaseTTSClient):
    def __init__(self, model: str, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("Missing dependency 'openai'. Install dependencies from requirements.txt.") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_path = Path(tmp.name)

        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "instructions": _tts_style_instruction(speaker),
            "response_format": "mp3",
        }
        try:
            with self.client.audio.speech.with_streaming_response.create(**payload) as response:
                response.stream_to_file(str(temp_path))
        except Exception as exc:
            # Older/alternate model configs might not accept instructions.
            if "instruction" in str(exc).lower():
                payload.pop("instructions", None)
                with self.client.audio.speech.with_streaming_response.create(**payload) as response:
                    response.stream_to_file(str(temp_path))
            else:
                temp_path.unlink(missing_ok=True)
                raise PipelineError(f"OpenAI TTS failed: {exc}") from exc

        try:
            return temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)


class ElevenLabsTTSClient(BaseTTSClient):
    def __init__(self, model: str, api_key: str) -> None:
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise PipelineError("Missing dependency 'requests'. Install dependencies from requirements.txt.") from exc

        self.model = model
        self.api_key = api_key

    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            raise PipelineError(f"ElevenLabs TTS failed: {exc}") from exc


def _recover_openai_response_text(response: object) -> str:
    parts: List[str] = []
    output = getattr(response, "output", None)
    if not output:
        return ""

    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in getattr(item, "content", []):
                c_type = getattr(content, "type", None)
                if c_type in {"output_text", "text"}:
                    text = getattr(content, "text", None)
                    if text:
                        parts.append(text)
        elif getattr(item, "text", None):
            parts.append(item.text)
    return "\n".join(parts).strip()


def _tts_style_instruction(speaker: str) -> str:
    if speaker == "Host A":
        return "Speak clearly in a warm, confident podcast style."
    return "Speak clearly in an energetic, curious podcast style."


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Generate a 2-person AI podcast from a research paper PDF.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")

    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic", "openrouter", "openai_compatible"],
        help="LLM provider.",
    )
    parser.add_argument("--llm-model", default="gpt-5.4-pro", help="LLM model name.")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL for openai_compatible provider, or override for openai/openrouter.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--dialogue-turns", type=int, default=24, help="Approximate number of host turns.")
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=70000,
        help="If paper text is longer, chunk+summarize before final dialogue generation.",
    )
    parser.add_argument("--chunk-size", type=int, default=12000, help="Chunk size used in summarize-then-compose flow.")

    parser.add_argument("--tts-provider", default="openai", choices=["openai", "elevenlabs"], help="TTS provider.")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts", help="TTS model name.")
    parser.add_argument("--host-a-voice", default="echo", help="Voice for Host A (male).")
    parser.add_argument("--host-b-voice", default="sage", help="Voice for Host B (female).")
    parser.add_argument("--output-format", default="mp3", choices=["mp3", "m4a"], help="Final podcast format.")

    ns = parser.parse_args()
    return Args(
        pdf=Path(ns.pdf),
        llm_provider=ns.llm_provider,
        llm_model=ns.llm_model,
        llm_base_url=ns.llm_base_url,
        tts_provider=ns.tts_provider,
        tts_model=ns.tts_model,
        host_a_voice=ns.host_a_voice,
        host_b_voice=ns.host_b_voice,
        output_format=ns.output_format,
        dialogue_turns=max(6, ns.dialogue_turns),
        temperature=ns.temperature,
        max_input_chars=max(5000, ns.max_input_chars),
        chunk_size=max(2000, ns.chunk_size),
    )


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise PipelineError("Missing dependency 'python-dotenv'. Install dependencies from requirements.txt.") from exc
    load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PipelineError(f"Required environment variable missing: {name}")
    return value


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def extract_pdf_text(pdf_path: Path) -> tuple[Optional[str], str]:
    if not pdf_path.exists():
        raise PipelineError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PipelineError(f"Input file must be a PDF: {pdf_path}")

    title: Optional[str] = None
    text = ""

    # First pass: pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        raw_title = getattr(reader.metadata, "title", None) if reader.metadata else None
        if raw_title and isinstance(raw_title, str):
            title = raw_title.strip() or None
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages).strip()
    except Exception:
        text = ""

    # Fallback pass: pdfplumber (often better for some layouts).
    if len(text) < 500:
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                pages = [(page.extract_text() or "") for page in pdf.pages]
            alt_text = "\n".join(pages).strip()
            if len(alt_text) > len(text):
                text = alt_text
        except Exception:
            pass

    if not text.strip():
        raise PipelineError("Could not extract readable text from the PDF.")

    return title, text


def choose_paper_name(pdf_path: Path, extracted_title: Optional[str], text: str) -> str:
    if extracted_title:
        low = extracted_title.lower().strip()
        if low not in {"untitled", "document", "microsoft word -"}:
            return sanitize_name(extracted_title)

    for line in text.splitlines()[:40]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if 20 <= len(candidate) <= 180 and not candidate.lower().startswith("arxiv"):
            return sanitize_name(candidate)

    return sanitize_name(pdf_path.stem)


def chunk_text(text: str, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: List[str] = []
    current = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            chunks.extend(_split_long_block(para, max_chars))
            continue

        projected = current_len + len(para) + 2
        if projected > max_chars:
            flush()
        current.append(para)
        current_len += len(para) + 2

    flush()
    return chunks or [text[:max_chars]]


def _split_long_block(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    cur = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            words = sentence.split()
            chunk_words: List[str] = []
            chunk_len = 0
            for word in words:
                add_len = len(word) + (1 if chunk_words else 0)
                if chunk_len + add_len > max_chars and chunk_words:
                    out.append(" ".join(chunk_words))
                    chunk_words = [word]
                    chunk_len = len(word)
                else:
                    chunk_words.append(word)
                    chunk_len += add_len
            if chunk_words:
                out.append(" ".join(chunk_words))
            continue

        candidate = f"{cur} {sentence}".strip() if cur else sentence
        if len(candidate) > max_chars and cur:
            out.append(cur)
            cur = sentence
        else:
            cur = candidate

    if cur:
        out.append(cur)
    return out


def summarize_if_needed(llm: BaseLLMClient, text: str, max_input_chars: int, chunk_size: int, temperature: float) -> str:
    if len(text) <= max_input_chars:
        return text

    chunks = chunk_text(text, chunk_size)
    summaries: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            f"Summarize chunk {i}/{len(chunks)} from a research paper.\n"
            "Focus on key claims, methods, results, and limitations.\n\n"
            f"{chunk}"
        )
        summary = llm.generate_text(
            system_prompt=CHUNK_SUMMARY_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=min(temperature, 0.3),
            max_output_tokens=1200,
        )
        summaries.append(f"Chunk {i} summary:\n{summary}")
    return "\n\n".join(summaries)


def generate_dialogue(
    llm: BaseLLMClient,
    *,
    paper_name: str,
    source_text: str,
    dialogue_turns: int,
    temperature: float,
) -> str:
    user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Generate approximately {dialogue_turns} turns total, alternating hosts.\n"
        "Use only this format:\n"
        "Host A: <text>\n"
        "Host B: <text>\n\n"
        "Source material:\n"
        f"{source_text}"
    )
    raw_dialogue = llm.generate_text(
        system_prompt=DIALOGUE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=12000,
    )
    turns = parse_dialogue_text(raw_dialogue)
    if len(turns) < 4:
        raise PipelineError("LLM returned too little dialogue to build podcast audio.")
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def parse_dialogue_text(dialogue_text: str) -> List[DialogueTurn]:
    turns: List[DialogueTurn] = []
    for raw_line in dialogue_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if match:
            speaker = "Host A" if "a" in match.group(1).lower() else "Host B"
            content = match.group(2).strip()
            turns.append(DialogueTurn(speaker=speaker, text=content))
        elif turns:
            # If model wrapped lines, attach continuation to previous speaker.
            turns[-1].text = f"{turns[-1].text} {line}".strip()

    turns = [turn for turn in turns if turn.text]
    if not turns:
        raise PipelineError(
            "Generated dialogue is not in parseable format. "
            "Expected lines starting with 'Host A:' and 'Host B:'."
        )
    return turns


def parse_dialogue_file(dialogue_path: Path) -> List[DialogueTurn]:
    if not dialogue_path.exists():
        raise PipelineError(f"Dialogue file not found: {dialogue_path}")

    turns: List[DialogueTurn] = []
    with dialogue_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = SPEAKER_RE.match(line)
            if match:
                speaker = "Host A" if "a" in match.group(1).lower() else "Host B"
                turns.append(DialogueTurn(speaker=speaker, text=match.group(2).strip()))
            elif turns:
                turns[-1].text = f"{turns[-1].text} {line}".strip()

    turns = [turn for turn in turns if turn.text]
    if not turns:
        raise PipelineError("Dialogue file is empty or not in 'Host A:/Host B:' format.")
    return turns


def split_for_tts(text: str, max_chars: int = 850) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    for block in _split_long_block(text, max_chars):
        if block:
            parts.append(block.strip())
    return parts


def build_llm_client(args: Args) -> BaseLLMClient:
    provider = args.llm_provider.lower()

    if provider == "openai":
        return OpenAIResponsesClient(
            model=args.llm_model,
            api_key=required_env("OPENAI_API_KEY"),
            base_url=args.llm_base_url,
        )

    if provider == "anthropic":
        return AnthropicClient(model=args.llm_model, api_key=required_env("ANTHROPIC_API_KEY"))

    if provider == "openrouter":
        base_url = args.llm_base_url or "https://openrouter.ai/api/v1"
        return OpenAIChatCompatibleClient(
            model=args.llm_model,
            api_key=required_env("OPENROUTER_API_KEY"),
            base_url=base_url,
        )

    if provider == "openai_compatible":
        base_url = args.llm_base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
        if not base_url:
            raise PipelineError(
                "openai_compatible provider requires --llm-base-url or OPENAI_COMPAT_BASE_URL."
            )
        api_key = os.getenv("OPENAI_COMPAT_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise PipelineError("Set OPENAI_COMPAT_API_KEY (or OPENAI_API_KEY) for openai_compatible provider.")
        return OpenAIChatCompatibleClient(model=args.llm_model, api_key=api_key, base_url=base_url)

    raise PipelineError(f"Unsupported llm-provider: {args.llm_provider}")


def build_tts_client(args: Args) -> BaseTTSClient:
    provider = args.tts_provider.lower()
    if provider == "openai":
        return OpenAITTSClient(model=args.tts_model, api_key=required_env("OPENAI_API_KEY"))
    if provider == "elevenlabs":
        return ElevenLabsTTSClient(model=args.tts_model, api_key=required_env("ELEVENLABS_API_KEY"))
    raise PipelineError(f"Unsupported tts-provider: {args.tts_provider}")


def synthesize_podcast_audio(
    *,
    turns: Iterable[DialogueTurn],
    tts_client: BaseTTSClient,
    host_a_voice: str,
    host_b_voice: str,
    output_path: Path,
) -> None:
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise PipelineError("Missing dependency 'pydub'. Install dependencies from requirements.txt.") from exc

    combined = AudioSegment.silent(duration=0)
    pause_between_segments = AudioSegment.silent(duration=120)
    pause_between_turns = AudioSegment.silent(duration=260)

    turn_count = 0
    for turn in turns:
        turn_count += 1
        voice = host_a_voice if turn.speaker == "Host A" else host_b_voice
        segments = split_for_tts(turn.text, max_chars=850)
        if not segments:
            continue
        for segment in segments:
            audio_bytes = tts_client.synthesize(text=segment, voice=voice, speaker=turn.speaker)
            try:
                clip = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            except Exception as exc:
                raise PipelineError(
                    "Failed to decode generated audio. Ensure ffmpeg is installed and available in PATH."
                ) from exc
            combined += clip + pause_between_segments
        combined += pause_between_turns

    if turn_count == 0 or len(combined) == 0:
        raise PipelineError("No audio clips were generated from dialogue.")

    export_format = "mp4" if output_path.suffix.lower() == ".m4a" else output_path.suffix.lstrip(".")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        combined.export(str(output_path), format=export_format)
    except Exception as exc:
        raise PipelineError("Audio export failed. Ensure ffmpeg is installed and available in PATH.") from exc


def save_text_file(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def move_files_into_paper_directory(*, paper_name: str, pdf_path: Path, text_path: Path, audio_path: Path) -> Path:
    target_dir = pdf_path.parent / paper_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for src in [pdf_path, text_path, audio_path]:
        dest = target_dir / src.name
        try:
            if src.resolve() == dest.resolve():
                continue
        except Exception:
            pass
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))

    return target_dir


def run(args: Args) -> Path:
    load_env()

    pdf_path = args.pdf.expanduser().resolve()
    extracted_title, paper_text = extract_pdf_text(pdf_path)
    paper_name = choose_paper_name(pdf_path, extracted_title, paper_text)

    llm_client = build_llm_client(args)
    source_for_dialogue = summarize_if_needed(
        llm_client, paper_text, args.max_input_chars, args.chunk_size, args.temperature
    )
    dialogue_text = generate_dialogue(
        llm_client,
        paper_name=paper_name,
        source_text=source_for_dialogue,
        dialogue_turns=args.dialogue_turns,
        temperature=args.temperature,
    )

    text_path = pdf_path.with_name(f"{paper_name}.txt")
    save_text_file(text_path, dialogue_text)

    # Requirement: parse the generated dialogue text file line-by-line for TTS.
    turns = parse_dialogue_file(text_path)

    tts_client = build_tts_client(args)
    audio_path = pdf_path.with_name(f"{paper_name}.{args.output_format}")
    synthesize_podcast_audio(
        turns=turns,
        tts_client=tts_client,
        host_a_voice=args.host_a_voice,
        host_b_voice=args.host_b_voice,
        output_path=audio_path,
    )

    final_dir = move_files_into_paper_directory(
        paper_name=paper_name,
        pdf_path=pdf_path,
        text_path=text_path,
        audio_path=audio_path,
    )
    return final_dir


def main() -> int:
    try:
        args = parse_args()
        out_dir = run(args)
        print(f"Completed. Files are in: {out_dir}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
