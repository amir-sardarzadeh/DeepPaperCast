#!/usr/bin/env python3
"""Emotion-tag middleware for Grok TTS transcripts."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)
STREAMING_MAX_TOKENS_THRESHOLD = 21333

ALLOWED_INLINE_TAGS = (
    "pause",
    "long-pause",
    "hum-tune",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "breath",
    "inhale",
    "exhale",
    "sigh",
)
ALLOWED_WRAPPING_TAGS = (
    "soft",
    "whisper",
    "loud",
    "build-intensity",
    "decrease-intensity",
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",
    "singing",
    "laugh-speak",
    "emphasis",
)
ALLOWED_INLINE_TAG_SET = {t.lower() for t in ALLOWED_INLINE_TAGS}
ALLOWED_WRAPPING_TAG_SET = {t.lower() for t in ALLOWED_WRAPPING_TAGS}

INLINE_TAG_RE = re.compile(r"\[(%s)\]" % "|".join(re.escape(t) for t in ALLOWED_INLINE_TAGS), flags=re.IGNORECASE)
OPEN_WRAP_TAG_RE = re.compile(r"<(%s)>" % "|".join(re.escape(t) for t in ALLOWED_WRAPPING_TAGS), flags=re.IGNORECASE)
CLOSE_WRAP_TAG_RE = re.compile(r"</(%s)>" % "|".join(re.escape(t) for t in ALLOWED_WRAPPING_TAGS), flags=re.IGNORECASE)
GENERIC_INLINE_TAG_RE = re.compile(r"\[([a-zA-Z-]+)\]")
GENERIC_WRAP_TAG_RE = re.compile(r"</?([a-zA-Z-]+)>")

EMOTION_SYSTEM_PROMPT = """You are an audio-tagging assistant for xAI Grok TTS.
Your job is to add expressive speech tags to an existing two-host transcript.

Hard rules:
- Do not change any spoken words.
- Do not remove words.
- Do not reorder words.
- Do not paraphrase.
- Keep every line and speaker in the exact same order.
- Keep each line starting with exactly "Host A:" or "Host B:".
- You may only insert Grok TTS emotion tags.
- Return transcript lines only. No comments, no markdown, no numbering.

Allowed inline tags:
[pause] [long-pause] [hum-tune] [laugh] [chuckle] [giggle] [cry]
[tsk] [tongue-click] [lip-smack] [breath] [inhale] [exhale] [sigh]

Allowed wrapping tags:
<soft> </soft> <whisper> </whisper> <loud> </loud>
<build-intensity> </build-intensity> <decrease-intensity> </decrease-intensity>
<higher-pitch> </higher-pitch> <lower-pitch> </lower-pitch>
<slow> </slow> <fast> </fast> <sing-song> </sing-song>
<singing> </singing> <laugh-speak> </laugh-speak> <emphasis> </emphasis>
"""


class EmotionError(RuntimeError):
    """Raised when emotion tagging fails."""


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise EmotionError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def _canonical_speaker(raw: str) -> str:
    return "Host A" if "a" in raw.lower() else "Host B"


def _normalize_dialogue_lines(raw_text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if match:
            speaker = _canonical_speaker(match.group(1))
            content = match.group(2).strip()
            if content:
                lines.append(f"{speaker}: {content}")
        elif lines:
            lines[-1] = f"{lines[-1]} {line}".strip()
    return lines


def _split_line(line: str) -> Tuple[str, str]:
    match = SPEAKER_RE.match(line.strip())
    if not match:
        raise EmotionError(f"Unparseable dialogue line: {line[:80]}")
    return _canonical_speaker(match.group(1)), match.group(2).strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_allowed_tags(text: str) -> str:
    stripped = INLINE_TAG_RE.sub("", text)
    stripped = OPEN_WRAP_TAG_RE.sub("", stripped)
    stripped = CLOSE_WRAP_TAG_RE.sub("", stripped)
    return _normalize_ws(stripped)


def _has_disallowed_tags(text: str) -> bool:
    for match in GENERIC_INLINE_TAG_RE.finditer(text):
        if match.group(1).strip().lower() not in ALLOWED_INLINE_TAG_SET:
            return True
    for match in GENERIC_WRAP_TAG_RE.finditer(text):
        if match.group(1).strip().lower() not in ALLOWED_WRAPPING_TAG_SET:
            return True
    return False


def _extract_message_text(message: object) -> str:
    parts: List[str] = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "\n".join(parts).strip()


def _send_anthropic(client: object, payload: Dict[str, object], force_stream: bool = False) -> object:
    max_tokens = int(payload.get("max_tokens", 0) or 0)
    use_stream = force_stream or max_tokens >= STREAMING_MAX_TOKENS_THRESHOLD
    if use_stream:
        with client.messages.stream(**payload) as stream:
            return stream.get_final_message()
    return client.messages.create(**payload)


def _is_opus_4_7_model(model: str) -> bool:
    return "claude-opus-4-7" in (model or "").lower()


def _generate_emotion_chunk(
    *,
    client: object,
    model: str,
    chunk_lines: List[str],
    thinking_budget_tokens: int,
) -> str:
    user_prompt = (
        "Add Grok TTS emotion tags to the transcript below.\n"
        "Keep all spoken words unchanged.\n\n"
        "Transcript:\n"
        f"{chr(10).join(chunk_lines)}"
    )

    approx_tokens = max(1024, len(user_prompt) // 3)
    max_output_tokens = min(16_384, max(2048, approx_tokens))
    thinking_budget = max(0, int(thinking_budget_tokens))
    if thinking_budget >= max_output_tokens:
        thinking_budget = max_output_tokens - 1
    if 0 < thinking_budget < 1024:
        thinking_budget = 0

    payload: Dict[str, object] = {
        "model": model,
        "system": EMOTION_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_output_tokens,
    }
    if thinking_budget > 0:
        if _is_opus_4_7_model(model):
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}
            payload["output_config"] = {"effort": "high"}
        else:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    else:
        if not _is_opus_4_7_model(model):
            payload["temperature"] = 0.2

    try:
        message = _send_anthropic(client, payload)
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc).lower()
        if "streaming is required" in error_text:
            try:
                message = _send_anthropic(client, payload, force_stream=True)
            except Exception as stream_exc:  # noqa: BLE001
                raise EmotionError(f"Emotion stage Anthropic API error: {stream_exc}") from stream_exc
        elif "thinking" in payload:
            payload.pop("thinking", None)
            if not _is_opus_4_7_model(model):
                payload["temperature"] = 0.2
            try:
                message = _send_anthropic(client, payload)
            except Exception as retry_exc:  # noqa: BLE001
                retry_text = str(retry_exc).lower()
                if "streaming is required" in retry_text:
                    try:
                        message = _send_anthropic(client, payload, force_stream=True)
                    except Exception as stream_retry_exc:  # noqa: BLE001
                        raise EmotionError(f"Emotion stage Anthropic API error: {stream_retry_exc}") from stream_retry_exc
                else:
                    raise EmotionError(f"Emotion stage Anthropic API error: {retry_exc}") from retry_exc
        else:
            raise EmotionError(f"Emotion stage Anthropic API error: {exc}") from exc

    text = _extract_message_text(message)
    if not text:
        raise EmotionError("Emotion stage returned no readable text.")
    return text


def _validate_chunk(original_lines: List[str], candidate_lines: List[str], logger: logging.Logger, chunk_idx: int) -> List[str]:
    if len(candidate_lines) != len(original_lines):
        logger.warning(
            "Emotion chunk %d returned %d lines, expected %d. Falling back to original lines for this chunk.",
            chunk_idx,
            len(candidate_lines),
            len(original_lines),
        )
        return list(original_lines)

    validated: List[str] = []
    replaced_count = 0
    for idx, (orig_line, cand_line) in enumerate(zip(original_lines, candidate_lines), start=1):
        try:
            orig_speaker, orig_text = _split_line(orig_line)
            cand_speaker, cand_text = _split_line(cand_line)
        except EmotionError:
            validated.append(orig_line)
            replaced_count += 1
            continue

        if cand_speaker != orig_speaker:
            validated.append(orig_line)
            replaced_count += 1
            continue
        if _has_disallowed_tags(cand_text):
            validated.append(orig_line)
            replaced_count += 1
            continue

        stripped_candidate = _strip_allowed_tags(cand_text)
        if stripped_candidate != _normalize_ws(orig_text):
            validated.append(orig_line)
            replaced_count += 1
            continue

        validated.append(f"{orig_speaker}: {cand_text}")

    if replaced_count > 0:
        logger.warning(
            "Emotion chunk %d replaced %d/%d lines with originals because validation failed.",
            chunk_idx,
            replaced_count,
            len(original_lines),
        )
    return validated


def add_emotion_tags_to_dialogue(
    *,
    dialogue_path: Path,
    output_dir: Path,
    api_keys: Dict[str, str],
    logger: logging.Logger,
    model: str = "claude-sonnet-4-6",
    thinking_budget_tokens: int = 4096,
    max_lines_per_chunk: int = 28,
) -> Path:
    """Create an emotion-tagged transcript copy validated against the original text."""
    if not dialogue_path.exists():
        raise EmotionError(f"Dialogue file not found for emotion stage: {dialogue_path}")

    raw = dialogue_path.read_text(encoding="utf-8")
    original_lines = _normalize_dialogue_lines(raw)
    if not original_lines:
        raise EmotionError("Emotion stage could not parse any dialogue lines.")

    try:
        import anthropic
    except ImportError as exc:
        raise EmotionError("Missing dependency 'anthropic'. Install requirements.txt.") from exc

    api_key = _get_api_key(api_keys, ["ANTHROPIC_API_KEY", "ANTHROPIC"])
    client = anthropic.Anthropic(api_key=api_key)

    max_lines_per_chunk = max(8, int(max_lines_per_chunk))
    final_lines: List[str] = []
    num_chunks = (len(original_lines) + max_lines_per_chunk - 1) // max_lines_per_chunk
    logger.info("Emotion stage enabled. Processing %d dialogue lines in %d chunk(s).", len(original_lines), num_chunks)

    for chunk_idx, start in enumerate(range(0, len(original_lines), max_lines_per_chunk), start=1):
        chunk = original_lines[start : start + max_lines_per_chunk]
        tagged_raw = _generate_emotion_chunk(
            client=client,
            model=model,
            chunk_lines=chunk,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        candidate_lines = _normalize_dialogue_lines(tagged_raw)
        validated_lines = _validate_chunk(chunk, candidate_lines, logger, chunk_idx)
        final_lines.extend(validated_lines)

    if len(final_lines) != len(original_lines):
        raise EmotionError("Emotion stage output line count mismatch after validation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dialogue_path.stem}_emotion.txt"
    output_path.write_text("\n".join(final_lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Emotion-tagged dialogue saved: %s", output_path)
    return output_path
