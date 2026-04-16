#!/usr/bin/env python3
"""LLM + PDF utilities for dialogue and explanation generation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

FORMULA_EXPLANATION_POLICY = """
- Formula Narration Rules:
- Never read an equation verbatim or as a stream of symbols.
- Never say raw notation such as "E equals one over r cubed", "sigma sub i", "x hat", or long chains of Greek letters unless a symbol name is absolutely necessary.
- When a formula appears, first state what physical, statistical, or algorithmic relationship it expresses.
- Then explain what each important term means in plain language.
- Then give an intuitive analogy, geometric picture, or real-world interpretation.
- Mention the equation number explicitly when one is available, but do not recite the equation after naming it.
- If the equation is dense, summarize the dependency or tradeoff it encodes instead of reading notation.
- Preferred style example: instead of saying "Equation 7 says E equals one over r cubed", say "Equation 7 says the field dies off extremely quickly with distance; if you move away from the source, the strength collapses roughly with the cube of distance, so small increases in distance produce a much weaker effect."
- Another preferred style example: instead of saying "y equals H x plus n", say "This is the standard measurement model: the system mixes the underlying signal through the channel, and then random noise gets added on top before we observe it."
"""

DEFAULT_DIALOGUE_SYSTEM_PROMPT = """You are a world-class podcast producer and scriptwriter.
Write a lively, engaging two-person podcast transcript between Host A and Host B about the provided academic paper.

Requirements:
- Keep the conversation highly engaging, conversational, and accessible.
- Host A is the structured guide; Host B is the insightful color commentator.
- Cover: problem, methodology, key findings, limitations, and practical implications.
- Formula Handling: If there are simple, foundational formulas, explain them intuitively using physical or visual analogies. Skip over highly complex, impenetrable math and just focus on the resulting concepts.
- Keep each speaking turn short and punchy (1-3 sentences maximum).
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions, or narrator notes. Do not read notation symbol-by-symbol; convert it into natural spoken explanation.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

HIGH_DETAIL_DIALOGUE_SYSTEM_PROMPT = f"""You are a world-class podcast producer and scriptwriter.
Write a highly comprehensive, extended-length, two-person podcast transcript between Host A and Host B about the provided academic paper.

Requirements:
- Maximize your output. Take the time to explain the background context and foundational concepts before diving into the paper's novel contributions.
- Detail the methodology step-by-step. Discuss why the authors chose this approach and alternatives.
- Host A acts as the expert guide. Host B acts as the highly curious learner asking probing follow-up questions.
- STRICT Formula Handling: You MUST explain all formulas presented in the paper. When discussing a formula, you must explicitly state its number as it appears in the text (for example: "If we look at Equation 4..."). Translate these complex equations into physical, visual, or relatable analogies so they make sense in an audio format. DO NOT just read the raw math symbols.
{FORMULA_EXPLANATION_POLICY}
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions, or narrator notes. Do not read notation symbol-by-symbol; convert it into natural spoken explanation.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

HIGH_DETAIL_PART1_SYSTEM_PROMPT = """You are a world-class podcast producer and scriptwriter creating a highly detailed, extended-length, two-person podcast.
You are writing Part 1 of 3.

Your specific task for Part 1:
Focus exclusively on the paper's introduction, the background context, the existing literature, and the core problem the authors are trying to solve.

Requirements:
- Maximize your output. Take your time setting up the stakes and foundational concepts. Assume the listener is intelligent but needs complex jargon unpacked.
- Host A acts as the expert guide. Host B acts as the highly curious learner asking probing follow-up questions.
- DO NOT dive into the deep methodology, equations, or final results yet. Leave those for later parts.
- DO NOT conclude or wrap up the episode. End Part 1 on a conversational cliffhanger or a transition leading into the methodology.
- Audio/TTS Constraints: DO NOT use emojis, markdown, bullet points, stage directions, or narrator notes.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

HIGH_DETAIL_PART2_SYSTEM_PROMPT = f"""You are a world-class podcast producer and scriptwriter. You are writing Part 2 of 3 of an ongoing podcast about the provided academic paper.

Below is the entire transcript of Part 1.

Your specific task for Part 2:
Read the transcript so far so you know exactly what has been covered. Then, seamlessly continue the conversation, focusing exclusively on the methodology, the architecture, and the complex math.

Requirements:
- DO NOT repeat introductions, greetings, or background information already covered in Part 1. Pick up the dialogue exactly where the previous text left off.
- Maximize your output. Detail the methodology step-by-step. Discuss why the authors chose this approach and what the alternatives were.
- STRICT Formula Handling: You MUST explain the core formulas presented in the paper. State the equation number explicitly (for example: "Looking at Equation 4..."). Translate these equations into relatable physical or visual analogies. DO NOT just read raw math symbols.
{FORMULA_EXPLANATION_POLICY}
- DO NOT cover the final experimental results or real-world implications yet.
- DO NOT conclude the episode. End on a transition leading toward the results.
- STRICT FORMAT: Every single line must begin with exactly "Host A:" or "Host B:".
"""

HIGH_DETAIL_PART3_SYSTEM_PROMPT = f"""You are a world-class podcast producer and scriptwriter. You are writing Part 3 of 3, the final segment of an ongoing podcast about the provided academic paper.

Below is the entire accumulated transcript of Parts 1 and 2.

Your specific task for Part 3:
Read the transcript so far to understand the full context. Then, seamlessly continue the conversation, focusing on the experimental results, the limitations/caveats, and the real-world implications of this research.

Requirements:
- DO NOT repeat the background or methodology. Pick up the dialogue exactly where Part 2 left off.
- Maximize your output to thoroughly unpack the data and what it actually means for the future of the field.
- Once the results and limitations have been thoroughly explored, guide the hosts to a natural, engaging outro and conclude the episode.
- If you briefly refer back to an earlier equation while interpreting the results, keep following the same meaning-first explanation style and do not recite notation.
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
ANTHROPIC_NON_STREAMING_MAX_TOKENS_FALLBACK = 21333


class LLMWriterError(RuntimeError):
    """Raised when PDF parsing or LLM generation fails."""


def load_api_keys(api_file: Path) -> Dict[str, str]:
    """Load API keys from api.txt style KEY=VALUE lines."""
    if not api_file.exists():
        raise LLMWriterError(f"API key file not found: {api_file}")

    keys: Dict[str, str] = {}
    for line_no, raw_line in enumerate(api_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LLMWriterError(f"Malformed line {line_no} in {api_file.name}: {line}")
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            raise LLMWriterError(f"Malformed line {line_no} in {api_file.name}: {line}")
        keys[key] = value
    return keys


def sanitize_name(name: str) -> str:
    """Sanitize string for filesystem-safe folder/file names."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def extract_pdf_text(pdf_path: Path, logger: logging.Logger) -> tuple[Optional[str], str]:
    """Extract title and text from PDF with pypdf and pdfplumber fallback."""
    if not pdf_path.exists():
        raise LLMWriterError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise LLMWriterError(f"Input is not a PDF: {pdf_path}")

    title: Optional[str] = None
    text = ""

    try:
        from pypdf import PdfReader

        logger.info("Extracting PDF text with pypdf: %s", pdf_path)
        reader = PdfReader(str(pdf_path))
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip() or None
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        logger.info("pypdf extraction complete. chars=%d", len(text))
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
            logger.info("pdfplumber extraction complete. chars=%d", len(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber extraction failed: %s", exc)

    if not text:
        raise LLMWriterError("Failed to extract readable text from PDF.")
    return title, text


def choose_paper_name(pdf_path: Path, extracted_title: Optional[str], text: str) -> str:
    """Choose paper title for file/folder naming."""
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
    """Estimate token count using tiktoken when available, with char fallback."""
    try:
        import tiktoken

        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except Exception:  # noqa: BLE001
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


class BaseLLMClient:
    """Base LLM client interface."""

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
        """Generate text response."""
        raise NotImplementedError


class OpenAIResponsesClient(BaseLLMClient):
    """OpenAI Responses API wrapper."""

    def __init__(self, *, model: str, api_key: str, base_url: Optional[str], logger: logging.Logger) -> None:
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
        self.logger.info("Calling OpenAI Responses API. model=%s", self.model)
        try:
            response = self.client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in str(exc).lower():
                payload.pop("temperature", None)
                response = self.client.responses.create(**payload)
            else:
                raise LLMWriterError(f"OpenAI API error: {exc}") from exc

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        recovered = _recover_openai_response_text(response)
        if recovered:
            return recovered.strip()
        raise LLMWriterError("OpenAI response did not contain readable text.")


class AnthropicMessagesClient(BaseLLMClient):
    """Anthropic Messages API wrapper using non-streaming responses."""

    def __init__(self, *, model: str, api_key: str, logger: logging.Logger) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMWriterError("Missing dependency 'anthropic'. Install requirements.txt.") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.logger = logger

    @staticmethod
    def _is_streaming_required_error(error_text: str) -> bool:
        return "streaming is required" in error_text.lower()

    @staticmethod
    def _extract_max_tokens_limit(error_text: str) -> Optional[int]:
        """
        Parse Anthropic max_tokens cap from errors like:
        "max_tokens: 168529 > 128000, which is the maximum allowed ..."
        """
        match = re.search(r"max_tokens:\s*\d+\s*>\s*(\d+)", error_text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _is_thinking_budget_relation_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return "max_tokens" in lowered and "thinking.budget_tokens" in lowered and "greater than" in lowered

    def _normalize_thinking_payload(self, payload: Dict[str, object]) -> None:
        """Ensure Anthropic thinking payload obeys budget constraints."""
        max_tokens = int(payload.get("max_tokens", 0) or 0)
        thinking_obj = payload.get("thinking")
        if not isinstance(thinking_obj, dict):
            return

        budget = int(thinking_obj.get("budget_tokens", 0) or 0)
        if budget >= max_tokens:
            budget = max_tokens - 1
        if budget < 1024:
            payload.pop("thinking", None)
            return
        thinking_obj["budget_tokens"] = budget
        payload["thinking"] = thinking_obj

    def _apply_non_streaming_cap_from_error(self, payload: Dict[str, object], error_text: str) -> bool:
        """Reduce max_tokens when Anthropic rejects non-streaming requests."""
        if not self._is_streaming_required_error(error_text):
            return False

        current = int(payload.get("max_tokens", 0) or 0)
        if current <= 1024:
            return False

        parsed_cap = self._extract_max_tokens_limit(error_text)
        if parsed_cap and parsed_cap < current:
            cap = parsed_cap
        elif current > ANTHROPIC_NON_STREAMING_MAX_TOKENS_FALLBACK:
            cap = ANTHROPIC_NON_STREAMING_MAX_TOKENS_FALLBACK
        else:
            cap = max(1024, current // 2)

        if cap >= current:
            return False

        payload["max_tokens"] = cap
        self._normalize_thinking_payload(payload)
        return True

    def _apply_token_cap_from_error(self, payload: Dict[str, object], error_text: str) -> bool:
        """
        Apply automatic payload fixes from token-related API errors.
        Returns True if payload was changed.
        """
        changed = False
        cap = self._extract_max_tokens_limit(error_text)
        if cap and int(payload.get("max_tokens", 0) or 0) > cap:
            payload["max_tokens"] = cap
            changed = True

        if self._apply_non_streaming_cap_from_error(payload, error_text):
            changed = True

        if self._is_thinking_budget_relation_error(error_text):
            self._normalize_thinking_payload(payload)
            changed = True

        thinking_obj = payload.get("thinking")
        if isinstance(thinking_obj, dict):
            pre_budget = int(thinking_obj.get("budget_tokens", 0) or 0)
            self._normalize_thinking_payload(payload)
            post_obj = payload.get("thinking")
            post_budget = int(post_obj.get("budget_tokens", 0) or 0) if isinstance(post_obj, dict) else 0
            if pre_budget != post_budget:
                changed = True

        return changed

    def _send(self, payload: Dict[str, object]) -> object:
        self.logger.debug("Using Anthropic non-streaming API. max_tokens=%s", payload.get("max_tokens"))
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
        max_output_tokens = max(256, int(max_output_tokens))
        thinking_budget_tokens = max(0, int(thinking_budget_tokens))
        if thinking_budget_tokens >= max_output_tokens:
            thinking_budget_tokens = max_output_tokens - 1
        if 0 < thinking_budget_tokens < 1024:
            thinking_budget_tokens = 0

        payload: Dict[str, object] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
        }

        use_thinking = enable_extended_thinking and thinking_budget_tokens > 0
        if use_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget_tokens)}
            if temperature != 1.0:
                self.logger.warning(
                    "Ignoring temperature=%s because Anthropic thinking mode is incompatible with custom temperature.",
                    temperature,
                )
        else:
            payload["temperature"] = temperature

        self.logger.info("Calling Anthropic Messages API. model=%s", self.model)
        try:
            message = self._send(payload)
        except Exception as exc:  # noqa: BLE001
            recovered = False
            error_text = str(exc)
            if self._apply_token_cap_from_error(payload, error_text):
                if "thinking" not in payload and "temperature" not in payload:
                    payload["temperature"] = temperature
                self.logger.warning(
                    "Anthropic request rejected by token constraints (%s). Retrying with max_tokens=%s%s.",
                    exc,
                    payload.get("max_tokens"),
                    f", thinking_budget={payload['thinking'].get('budget_tokens')}" if isinstance(payload.get("thinking"), dict) else "",
                )
                try:
                    message = self._send(payload)
                    recovered = True
                except Exception as adjusted_exc:  # noqa: BLE001
                    error_text = str(adjusted_exc)
                    exc = adjusted_exc

            if recovered:
                pass
            elif "thinking" in payload:
                self.logger.warning("Anthropic thinking request failed (%s). Retrying without thinking.", exc)
                payload.pop("thinking", None)
                payload["temperature"] = temperature
                try:
                    message = self._send(payload)
                except Exception as retry_exc:  # noqa: BLE001
                    retry_text = str(retry_exc)
                    if self._apply_token_cap_from_error(payload, retry_text):
                        self.logger.warning(
                            "Anthropic non-thinking retry also exceeded token limit. Retrying with max_tokens=%s.",
                            payload.get("max_tokens"),
                        )
                        try:
                            message = self._send(payload)
                        except Exception as final_retry_exc:  # noqa: BLE001
                            raise LLMWriterError(f"Anthropic API error: {final_retry_exc}") from final_retry_exc
                    else:
                        raise LLMWriterError(f"Anthropic API error: {retry_exc}") from retry_exc
            else:
                raise LLMWriterError(f"Anthropic API error: {exc}") from exc

        text_blocks: List[str] = []
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_blocks.append(block.text)
        if text_blocks:
            return "\n".join(text_blocks).strip()
        raise LLMWriterError("Anthropic response did not include readable text.")


def _recover_openai_response_text(response: object) -> str:
    """Recover text from lower-level OpenAI response structure."""
    parts: List[str] = []
    output = getattr(response, "output", None)
    if not output:
        return ""
    for item in output:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                ctype = getattr(content, "type", None)
                if ctype in {"output_text", "text"} and getattr(content, "text", None):
                    parts.append(content.text)
        elif getattr(item, "text", None):
            parts.append(item.text)
    return "\n".join(parts).strip()


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise LLMWriterError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def _build_llm_client(
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
    raise LLMWriterError(f"Unsupported LLM provider: {provider}")


def _split_long_block(text: str, max_chars: int) -> List[str]:
    """Split long text preserving sentence boundaries when possible."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            bucket: List[str] = []
            size = 0
            for word in words:
                add = len(word) + (1 if bucket else 0)
                if size + add > max_chars and bucket:
                    chunks.append(" ".join(bucket))
                    bucket = [word]
                    size = len(word)
                else:
                    bucket.append(word)
                    size += add
            if bucket:
                chunks.append(" ".join(bucket))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def maybe_summarize_long_text(
    *,
    llm_client: BaseLLMClient,
    text: str,
    max_input_chars: int,
    chunk_size: int,
    logger: logging.Logger,
) -> str:
    """Chunk-summarize very long text to avoid context overflow."""
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

    logger.info("Input text too long (%d chars). Summarizing %d chunks.", len(text), len(chunks))
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


def _normalize_dialogue(raw: str) -> List[str]:
    """Normalize LLM output into strict Host A/B lines."""
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
        raise LLMWriterError("No parseable Host A / Host B lines were produced by the LLM.")
    return lines


def _generate_high_detail_macro_chunks(
    *,
    llm_client: BaseLLMClient,
    paper_name: str,
    llm_input: str,
    dialogue_turns: int,
    temperature: float,
    max_output_tokens: int,
    thinking_budget_tokens: int,
    logger: logging.Logger,
) -> List[str]:
    """
    Generate High-detail script using a 3-part macro-chunking sequence.
    Part 1: Context/problem.
    Part 2: Methodology/math (continues from Part 1 transcript).
    Part 3: Results/implications/outro (continues from Parts 1+2 transcript).
    """
    per_part_max_output = max(256, int(max_output_tokens) // 3)
    per_part_thinking = max(0, int(thinking_budget_tokens) // 3)
    if per_part_thinking >= per_part_max_output:
        per_part_thinking = per_part_max_output - 1
    if 0 < per_part_thinking < 1024:
        per_part_thinking = 0
    per_part_turn_target = max(8, int(dialogue_turns) // 3)

    logger.info(
        "High mode macro-chunking enabled. per_part_max_output=%d per_part_thinking=%d target_turns_per_part~%d",
        per_part_max_output,
        per_part_thinking,
        per_part_turn_target,
    )

    part1_user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Target speaking turns for this part: approximately {per_part_turn_target}.\n\n"
        "Paper content:\n"
        f"{llm_input}"
    )
    part1_raw = llm_client.generate_text(
        system_prompt=HIGH_DETAIL_PART1_SYSTEM_PROMPT,
        user_prompt=part1_user_prompt,
        temperature=temperature,
        max_output_tokens=per_part_max_output,
        thinking_budget_tokens=per_part_thinking,
        enable_extended_thinking=(per_part_thinking > 0),
    )
    part1_lines = _normalize_dialogue(part1_raw)
    transcript_so_far = "\n".join(part1_lines)

    part2_user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Target speaking turns for this part: approximately {per_part_turn_target}.\n\n"
        "Continue from the transcript below.\n\n"
        "[FULL_TRANSCRIPT_SO_FAR]\n"
        f"{transcript_so_far}\n\n"
        "Critical reminder for this part: when you reach equations, explain them in meaning-first audio language. Do not read formulas or symbol strings aloud.\n\n"
        "Paper content:\n"
        f"{llm_input}"
    )
    part2_raw = llm_client.generate_text(
        system_prompt=HIGH_DETAIL_PART2_SYSTEM_PROMPT,
        user_prompt=part2_user_prompt,
        temperature=temperature,
        max_output_tokens=per_part_max_output,
        thinking_budget_tokens=per_part_thinking,
        enable_extended_thinking=(per_part_thinking > 0),
    )
    part2_lines = _normalize_dialogue(part2_raw)
    transcript_so_far = "\n".join(part1_lines + part2_lines)

    part3_user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Target speaking turns for this part: approximately {per_part_turn_target}.\n\n"
        "Continue from the transcript below.\n\n"
        "[FULL_TRANSCRIPT_SO_FAR]\n"
        f"{transcript_so_far}\n\n"
        "Paper content:\n"
        f"{llm_input}"
    )
    part3_raw = llm_client.generate_text(
        system_prompt=HIGH_DETAIL_PART3_SYSTEM_PROMPT,
        user_prompt=part3_user_prompt,
        temperature=temperature,
        max_output_tokens=per_part_max_output,
        thinking_budget_tokens=per_part_thinking,
        enable_extended_thinking=(per_part_thinking > 0),
    )
    part3_lines = _normalize_dialogue(part3_raw)

    combined = part1_lines + part2_lines + part3_lines
    logger.info(
        "High mode macro-chunking complete. Lines: part1=%d part2=%d part3=%d total=%d",
        len(part1_lines),
        len(part2_lines),
        len(part3_lines),
        len(combined),
    )
    return combined


def _generate_medium_detail_dialogue(
    *,
    llm_client: BaseLLMClient,
    paper_name: str,
    llm_input: str,
    dialogue_turns: int,
    temperature: float,
    max_output_tokens: int,
    thinking_budget_tokens: int,
) -> List[str]:
    """
    Generate Medium-detail dialogue using the previous single-pass high-detail flow.
    This preserves the old High behavior while current High uses macro-chunking.
    """
    user_prompt = (
        f"Paper title: {paper_name}\n\n"
        f"Generate approximately {dialogue_turns} turns total.\n"
        "Allowed output format only:\n"
        "Host A: <text>\n"
        "Host B: <text>\n\n"
        "Paper content:\n"
        f"{llm_input}"
    )

    raw = llm_client.generate_text(
        system_prompt=HIGH_DETAIL_DIALOGUE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max(256, int(max_output_tokens)),
        thinking_budget_tokens=max(0, int(thinking_budget_tokens)),
        enable_extended_thinking=(thinking_budget_tokens > 0),
    )
    lines = _normalize_dialogue(raw)

    if len(lines) < max(24, dialogue_turns):
        retry_prompt = (
            f"{user_prompt}\n\n"
            "Quality update:\n"
            "- Expand background context and methods in more depth.\n"
            "- Explain all equations with explicit equation numbers when present.\n"
            "- Never read formulas aloud as symbol strings; explain what each equation means in plain language first.\n"
            "- Keep strict Host A / Host B formatting."
        )
        retry_raw = llm_client.generate_text(
            system_prompt=HIGH_DETAIL_DIALOGUE_SYSTEM_PROMPT,
            user_prompt=retry_prompt,
            temperature=temperature,
            max_output_tokens=max(256, int(max_output_tokens)),
            thinking_budget_tokens=max(0, int(thinking_budget_tokens)),
            enable_extended_thinking=(thinking_budget_tokens > 0),
        )
        retry_lines = _normalize_dialogue(retry_raw)
        if len(retry_lines) > len(lines):
            lines = retry_lines

    return lines


def generate_dialogue_text(
    *,
    paper_name: str,
    paper_text: str,
    api_keys: Dict[str, str],
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    detail_level: str,
    max_output_tokens: int,
    thinking_budget_tokens: int,
    dialogue_turns: int,
    temperature: float,
    max_input_chars: int,
    chunk_size: int,
    logger: logging.Logger,
) -> str:
    """Generate normalized Host A/B dialogue text."""
    detail = detail_level.strip().lower()
    if detail not in {"default", "medium", "high"}:
        raise LLMWriterError("DETAIL_LEVEL must be 'Default', 'Medium', or 'High'.")

    client = _build_llm_client(
        provider=llm_provider,
        model=llm_model,
        api_keys=api_keys,
        llm_base_url=llm_base_url,
        logger=logger,
    )
    llm_input = maybe_summarize_long_text(
        llm_client=client,
        text=paper_text,
        max_input_chars=max_input_chars,
        chunk_size=chunk_size,
        logger=logger,
    )

    if detail == "high":
        lines = _generate_high_detail_macro_chunks(
            llm_client=client,
            paper_name=paper_name,
            llm_input=llm_input,
            dialogue_turns=dialogue_turns,
            temperature=temperature,
            max_output_tokens=max(256, int(max_output_tokens)),
            thinking_budget_tokens=max(0, int(thinking_budget_tokens)),
            logger=logger,
        )
    elif detail == "medium":
        lines = _generate_medium_detail_dialogue(
            llm_client=client,
            paper_name=paper_name,
            llm_input=llm_input,
            dialogue_turns=dialogue_turns,
            temperature=temperature,
            max_output_tokens=max(256, int(max_output_tokens)),
            thinking_budget_tokens=max(0, int(thinking_budget_tokens)),
        )
    else:
        user_prompt = (
            f"Paper title: {paper_name}\n\n"
            f"Generate approximately {dialogue_turns} turns total.\n"
            "Allowed output format only:\n"
            "Host A: <text>\n"
            "Host B: <text>\n\n"
            "Paper content:\n"
            f"{llm_input}"
        )
        raw = client.generate_text(
            system_prompt=DEFAULT_DIALOGUE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max(256, int(max_output_tokens)),
            thinking_budget_tokens=max(0, int(thinking_budget_tokens)),
            enable_extended_thinking=(thinking_budget_tokens > 0),
        )
        lines = _normalize_dialogue(raw)

    return "\n".join(lines)


def save_dialogue_text(*, dialogue_text: str, output_dir: Path, paper_name: str) -> Path:
    """Save generated dialogue text to output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{paper_name}.txt"
    path.write_text(dialogue_text.rstrip() + "\n", encoding="utf-8")
    return path


def generate_dialogue_file_from_text(
    *,
    paper_name: str,
    paper_text: str,
    output_dir: Path,
    api_keys: Dict[str, str],
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    detail_level: str,
    max_output_tokens: int,
    thinking_budget_tokens: int,
    dialogue_turns: int,
    temperature: float,
    max_input_chars: int,
    chunk_size: int,
    logger: logging.Logger,
) -> Path:
    """Generate and save dialogue file from already extracted paper text."""
    dialogue_text = generate_dialogue_text(
        paper_name=paper_name,
        paper_text=paper_text,
        api_keys=api_keys,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        detail_level=detail_level,
        max_output_tokens=max_output_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        dialogue_turns=dialogue_turns,
        temperature=temperature,
        max_input_chars=max_input_chars,
        chunk_size=chunk_size,
        logger=logger,
    )
    path = save_dialogue_text(dialogue_text=dialogue_text, output_dir=output_dir, paper_name=paper_name)
    logger.info("Dialogue file saved: %s", path)
    return path
