"""PDF extraction + LLM dialogue writer for the podcast generator pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DIALOGUE_SYSTEM_PROMPT = """You are a world-class podcast producer and scriptwriter.
Write a lively, engaging two-person podcast transcript between Host A and Host B about the provided academic paper.
This should feel like a deep-dive educational episode, not a quick summary.

Host Dynamics:
- Host A is the structured guide, introducing the paper and driving the narrative forward.
- Host B is the insightful color commentator, asking clarifying questions, summarizing complex points, and providing relatable analogies.

Requirements:
- Keep the conversation highly engaging, conversational, and accessible.
- Explain highly technical ideas and complex math conceptually. DO NOT read raw equations, Greek letters, or code blocks (Only read them if they are clear and not long). Translate math into physical or visual analogies.
- Cover in depth: background context, the core problem, prior approaches, the paper's methodology, experiments, key findings, limitations, and practical real-world implications.
- Include concrete details from the paper (datasets, baselines, metrics, or result magnitudes) whenever available.
- Keep most speaking turns in the 2-5 sentence range, with occasional longer turns when unpacking complex concepts.
- Do not over-compress. Prioritize clear explanation of how the paper works step-by-step.
- This is a detailed-only format: never provide a short overview version.
- Style for voice delivery: use bright, intellectually curious language with warm enthusiasm. Keep banter brisk, but slow down when introducing a dense concept or the main takeaway.
- Add occasional transition hooks and analogy setups, for example: "But here's the crazy part," or "Think about it like this," where natural.
- Use punctuation that helps spoken rhythm: brief conversational beats around dense terms and just before analogies.
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions (e.g., [laughs]), or narrator notes. Spell out symbols (e.g., write "percent" instead of "%").
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
- Start with Host A welcoming the listener and alternate naturally.
"""

CHUNK_SUMMARY_SYSTEM_PROMPT = """You are an expert research assistant preparing notes for a podcast host. 
Summarize the provided section of an academic paper.

Return concise plain text structured strictly with the following headings:
1) Core Claim: (What is the main argument or finding here?)
2) Methods/Approach: (How did they do it? Keep it conceptual.)
3) Key Evidence: (The most important data points or results.)
4) Caveats/Limits: (What are the blind spots or assumptions?)
5) Real-World Significance: (Why does this actually matter?)
6) Jargon & Acronyms: (List any complex terms used here that need simple definitions.)
7) Suggested Analogy: (Provide one simple, everyday metaphor to explain the most complex idea in this section.)
"""

SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class DialogueResult:
    """Container for generated dialogue file output."""

    paper_name: str
    dialogue_path: Path


class LLMWriterError(RuntimeError):
    """Raised when PDF parsing or dialogue generation fails."""


class BaseLLMClient:
    """Base interface for provider-specific LLM clients."""

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        enable_extended_thinking: bool = False,
    ) -> str:
        """Generate and return plain text for the given prompt."""
        raise NotImplementedError


class OpenAIResponsesClient(BaseLLMClient):
    """OpenAI Responses API client wrapper."""

    def __init__(self, *, model: str, api_key: str, base_url: Optional[str], logger: logging.Logger) -> None:
        """Initialize an OpenAI Responses client."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMWriterError("Missing dependency 'openai'. Install requirements.txt.") from exc

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
        enable_extended_thinking: bool = False,
    ) -> str:
        """Send prompt to OpenAI Responses API and return text."""
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        self.logger.debug("Calling OpenAI Responses API (model=%s).", self.model)
        try:
            response = self.client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in str(exc).lower():
                self.logger.warning("Model rejected temperature; retrying without it.")
                payload.pop("temperature", None)
                response = self.client.responses.create(**payload)
            else:
                raise LLMWriterError(f"OpenAI Responses API call failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        recovered = _recover_openai_response_text(response)
        if recovered:
            return recovered.strip()
        raise LLMWriterError("OpenAI response did not contain readable text.")


class OpenAIChatCompatibleClient(BaseLLMClient):
    """Client wrapper for OpenAI-compatible Chat Completions endpoints."""

    def __init__(self, *, model: str, api_key: str, base_url: str, logger: logging.Logger) -> None:
        """Initialize an OpenAI-compatible chat client."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMWriterError("Missing dependency 'openai'. Install requirements.txt.") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.logger = logger

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        enable_extended_thinking: bool = False,
    ) -> str:
        """Send prompt to an OpenAI-compatible chat endpoint and return text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        self.logger.debug("Calling OpenAI-compatible chat API (model=%s).", self.model)
        try:
            completion = self.client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in str(exc).lower():
                self.logger.warning("Model rejected temperature; retrying without it.")
                payload.pop("temperature", None)
                completion = self.client.chat.completions.create(**payload)
            else:
                raise LLMWriterError(f"OpenAI-compatible chat call failed: {exc}") from exc

        message = completion.choices[0].message.content
        if isinstance(message, str):
            return message.strip()
        if isinstance(message, list):
            parts = []
            for block in message:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()
        raise LLMWriterError("Chat completion response did not include readable text.")


class AnthropicMessagesClient(BaseLLMClient):
    """Anthropic Messages API client wrapper."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        logger: logging.Logger,
        extended_thinking_enabled: bool = False,
        thinking_budget_tokens: int = 12000,
    ) -> None:
        """Initialize an Anthropic client."""
        try:
            import anthropic
        except ImportError as exc:
            raise LLMWriterError("Missing dependency 'anthropic'. Install requirements.txt.") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.logger = logger
        self.extended_thinking_enabled = extended_thinking_enabled
        self.thinking_budget_tokens = max(1024, thinking_budget_tokens)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        enable_extended_thinking: bool = False,
    ) -> str:
        """Send prompt to Anthropic Messages API and return text."""
        self.logger.debug("Calling Anthropic Messages API (model=%s).", self.model)
        payload = {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.extended_thinking_enabled and enable_extended_thinking:
            max_budget_for_request = max_output_tokens - 1
            budget = min(self.thinking_budget_tokens, max_budget_for_request)
            if budget >= 1024:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                self.logger.info("Anthropic extended thinking enabled. budget_tokens=%d", budget)
                if temperature != 1.0:
                    self.logger.warning(
                        "Ignoring temperature=%s because Anthropic extended thinking is not compatible with "
                        "custom temperature settings.",
                        temperature,
                    )
            else:
                self.logger.warning(
                    "Skipping extended thinking: max_tokens=%d leaves insufficient budget (<1024).",
                    max_output_tokens,
                )
                payload["temperature"] = temperature
        else:
            payload["temperature"] = temperature
        try:
            message = self.client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001
            raise LLMWriterError(f"Anthropic Messages API call failed: {exc}") from exc

        parts: List[str] = []
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                parts.append(block.text)
        if parts:
            return "\n".join(parts).strip()
        raise LLMWriterError("Anthropic response did not include readable text.")


def _recover_openai_response_text(response: object) -> str:
    """Recover text from lower-level Responses API output structure."""
    parts: List[str] = []
    output = getattr(response, "output", None)
    if not output:
        return ""
    for item in output:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                content_type = getattr(content, "type", None)
                if content_type in {"output_text", "text"} and getattr(content, "text", None):
                    parts.append(content.text)
        elif getattr(item, "text", None):
            parts.append(item.text)
    return "\n".join(parts).strip()


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    """Resolve an API key from aliases; raise an error if none exists."""
    for alias in aliases:
        key = api_keys.get(alias.upper(), "").strip()
        if key:
            return key
    raise LLMWriterError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def _build_llm_client(
    *,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_keys: Dict[str, str],
    logger: logging.Logger,
    anthropic_extended_thinking: bool = False,
    anthropic_thinking_budget_tokens: int = 12000,
) -> BaseLLMClient:
    """Construct the provider-specific LLM client."""
    p = provider.lower()
    if p == "openai":
        return OpenAIResponsesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENAI_API_KEY", "OPENAI"]),
            base_url=base_url,
            logger=logger,
        )
    if p == "anthropic":
        return AnthropicMessagesClient(
            model=model,
            api_key=_get_api_key(api_keys, ["ANTHROPIC_API_KEY", "ANTHROPIC"]),
            logger=logger,
            extended_thinking_enabled=anthropic_extended_thinking,
            thinking_budget_tokens=anthropic_thinking_budget_tokens,
        )
    if p == "openrouter":
        return OpenAIChatCompatibleClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENROUTER_API_KEY", "OPENROUTER"]),
            base_url=base_url or "https://openrouter.ai/api/v1",
            logger=logger,
        )
    if p == "openai_compatible":
        resolved_base = base_url or api_keys.get("OPENAI_COMPAT_BASE_URL", "").strip()
        if not resolved_base:
            raise LLMWriterError(
                "openai_compatible requires --llm-base-url or OPENAI_COMPAT_BASE_URL in api.txt."
            )
        return OpenAIChatCompatibleClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT", "OPENAI_API_KEY", "OPENAI"]),
            base_url=resolved_base,
            logger=logger,
        )
    raise LLMWriterError(f"Unsupported LLM provider: {provider}")


def _sanitize_name(name: str) -> str:
    """Sanitize a file/folder name for cross-platform filesystem usage."""
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def extract_pdf_text(pdf_path: Path, logger: logging.Logger) -> tuple[Optional[str], str]:
    """Extract PDF title metadata and text content with fallback parsing."""
    if not pdf_path.exists():
        raise LLMWriterError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise LLMWriterError(f"Input file is not a PDF: {pdf_path}")

    title: Optional[str] = None
    text = ""

    try:
        from pypdf import PdfReader

        logger.info("Extracting text via pypdf.")
        reader = PdfReader(str(pdf_path))
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip() or None
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        logger.info("pypdf extraction complete. Characters extracted: %d", len(text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf extraction failed: %s", exc)

    if len(text) < 500:
        try:
            import pdfplumber

            logger.info("Falling back to pdfplumber extraction.")
            with pdfplumber.open(str(pdf_path)) as pdf:
                alt_text = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
            if len(alt_text) > len(text):
                text = alt_text
            logger.info("pdfplumber extraction complete. Characters extracted: %d", len(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber extraction failed: %s", exc)

    if not text:
        raise LLMWriterError("Failed to extract readable text from the PDF.")
    return title, text


def choose_paper_name(pdf_path: Path, extracted_title: Optional[str], text: str) -> str:
    """Choose a filesystem-safe paper name from title metadata, first page, or filename."""
    if extracted_title:
        lowered = extracted_title.lower().strip()
        if lowered not in {"untitled", "document", "microsoft word -"}:
            return _sanitize_name(extracted_title)

    for line in text.splitlines()[:40]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if 20 <= len(candidate) <= 180 and not candidate.lower().startswith("arxiv"):
            return _sanitize_name(candidate)
    return _sanitize_name(pdf_path.stem)


def _split_long_block(text: str, max_chars: int) -> List[str]:
    """Split long text into <= max_chars chunks while preserving sentence boundaries when possible."""
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
            chunk_words: List[str] = []
            chunk_len = 0
            for word in words:
                add = len(word) + (1 if chunk_words else 0)
                if chunk_len + add > max_chars and chunk_words:
                    out.append(" ".join(chunk_words))
                    chunk_words = [word]
                    chunk_len = len(word)
                else:
                    chunk_words.append(word)
                    chunk_len += add
            if chunk_words:
                out.append(" ".join(chunk_words))
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


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Chunk extracted paper text into paragraph-based segments."""
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            chunks.extend(_split_long_block(paragraph, max_chars))
            continue
        projected = current_len + len(paragraph) + 2
        if projected > max_chars:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + 2
    flush()
    return chunks or [text[:max_chars]]


def _summarize_if_needed(
    *,
    llm_client: BaseLLMClient,
    text: str,
    max_input_chars: int,
    chunk_size: int,
    temperature: float,
    logger: logging.Logger,
) -> str:
    """Summarize long paper text in chunks if it exceeds model input budget."""
    if len(text) <= max_input_chars:
        logger.info("Paper text under input threshold; sending full text to LLM.")
        return text

    chunks = _chunk_text(text, chunk_size)
    logger.info("Paper text too long (%d chars). Summarizing %d chunks first.", len(text), len(chunks))
    summaries: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        logger.info("Summarizing chunk %d/%d.", index, len(chunks))
        prompt = (
            f"Summarize chunk {index}/{len(chunks)} from a research paper.\n"
            "Focus on claims, methods, evidence, and limits.\n\n"
            f"{chunk}"
        )
        summary = llm_client.generate_text(
            system_prompt=CHUNK_SUMMARY_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=min(temperature, 0.3),
            max_output_tokens=1200,
            enable_extended_thinking=False,
        )
        summaries.append(f"Chunk {index} summary:\n{summary}")
    logger.info("Chunk summarization complete.")
    return "\n\n".join(summaries)


def _parse_and_normalize_dialogue(dialogue_text: str) -> List[str]:
    """Normalize LLM output into strict 'Host A:'/'Host B:' line format."""
    lines: List[str] = []
    for raw_line in dialogue_text.splitlines():
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
        raise LLMWriterError("LLM response could not be parsed into Host A/Host B dialogue lines.")
    return lines


def _dialogue_word_count(lines: List[str]) -> int:
    """Count words in normalized dialogue lines."""
    return len(re.findall(r"\b\w+\b", "\n".join(lines)))


def _generate_dialogue_text(
    *,
    llm_client: BaseLLMClient,
    paper_name: str,
    source_text: str,
    dialogue_turns: int,
    temperature: float,
    logger: logging.Logger,
) -> str:
    """Generate dialogue text from paper content using the selected LLM client."""
    logger.info("Generating final dialogue with %d target turns.", dialogue_turns)
    base_user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Generate approximately {dialogue_turns} turns total.\n"
        "Allowed output format only:\n"
        "Host A: <text>\n"
        "Host B: <text>\n\n"
        "Paper content:\n"
        f"{source_text}"
    )
    raw = llm_client.generate_text(
        system_prompt=DIALOGUE_SYSTEM_PROMPT,
        user_prompt=base_user_prompt,
        temperature=temperature,
        max_output_tokens=18000,
        enable_extended_thinking=True,
    )
    lines = _parse_and_normalize_dialogue(raw)
    if len(lines) < 4:
        raise LLMWriterError("Generated dialogue is too short to build podcast audio.")

    min_lines = max(24, dialogue_turns)
    min_words = max(1500, dialogue_turns * 45)
    words = _dialogue_word_count(lines)
    if len(lines) < min_lines or words < min_words:
        logger.warning(
            "Dialogue looks too concise (lines=%d, words=%d). Regenerating with stronger depth constraints.",
            len(lines),
            words,
        )
        depth_prompt = (
            f"{base_user_prompt}\n\n"
            "Quality bar update:\n"
            "- Expand background context and method details.\n"
            "- Explain how the approach works step-by-step.\n"
            "- Include concrete experimental details and limitations.\n"
            "- Keep the format strictly Host A/Host B lines.\n"
            "- Keep most turns at 3-6 sentences, unless a shorter interjection is needed for natural flow.\n"
            f"- Minimum target: at least {min_lines} lines and about {min_words} words total."
        )
        raw_retry = llm_client.generate_text(
            system_prompt=DIALOGUE_SYSTEM_PROMPT,
            user_prompt=depth_prompt,
            temperature=temperature,
            max_output_tokens=20000,
            enable_extended_thinking=True,
        )
        retry_lines = _parse_and_normalize_dialogue(raw_retry)
        retry_words = _dialogue_word_count(retry_lines)
        if len(retry_lines) > len(lines) or retry_words > words:
            lines = retry_lines
            words = retry_words

    if len(lines) < min_lines or words < min_words:
        raise LLMWriterError(
            f"Detailed-only mode failed: dialogue still too short (lines={len(lines)}, words={words}). "
            f"Expected at least ~{min_lines} lines and ~{min_words} words."
        )

    logger.info("Dialogue generation complete. Parsed lines: %d, words: %d", len(lines), words)
    return "\n".join(lines)


def generate_dialogue_file_from_pdf(
    *,
    pdf_path: Path,
    api_keys: Dict[str, str],
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    temperature: float,
    dialogue_turns: int,
    max_input_chars: int,
    chunk_size: int,
    anthropic_extended_thinking: bool,
    anthropic_thinking_budget_tokens: int,
    output_dir: Path,
    logger: logging.Logger,
) -> DialogueResult:
    """Extract PDF text, generate Host A/B dialogue with LLM, and save the dialogue file."""
    title, extracted_text = extract_pdf_text(pdf_path, logger)
    paper_name = choose_paper_name(pdf_path, title, extracted_text)
    logger.info("Resolved paper name: %s", paper_name)

    llm_client = _build_llm_client(
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_keys=api_keys,
        logger=logger,
        anthropic_extended_thinking=anthropic_extended_thinking,
        anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
    )
    logger.info("LLM client ready. Provider=%s, Model=%s", llm_provider, llm_model)

    llm_input = _summarize_if_needed(
        llm_client=llm_client,
        text=extracted_text,
        max_input_chars=max_input_chars,
        chunk_size=chunk_size,
        temperature=temperature,
        logger=logger,
    )
    dialogue_text = _generate_dialogue_text(
        llm_client=llm_client,
        paper_name=paper_name,
        source_text=llm_input,
        dialogue_turns=dialogue_turns,
        temperature=temperature,
        logger=logger,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dialogue_path = output_dir / f"{paper_name}.txt"
    dialogue_path.write_text(dialogue_text + "\n", encoding="utf-8")
    logger.info("Dialogue text file saved: %s", dialogue_path)

    return DialogueResult(paper_name=paper_name, dialogue_path=dialogue_path)
