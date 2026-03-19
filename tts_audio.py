"""TTS generation utilities with persisted voice segment fail-safes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from pydub import AudioSegment

SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)
TARGET_SAMPLE_RATE_HZ = 48_000
TARGET_MP3_BITRATE = "320k"


class TTSAudioError(RuntimeError):
    """Raised when TTS generation or audio stitching fails."""


@dataclass
class DialogueTurn:
    """Represents one parsed dialogue turn."""

    speaker: str
    text: str


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise TTSAudioError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def parse_dialogue_file(dialogue_path: Path, logger: logging.Logger) -> List[DialogueTurn]:
    """Parse Host A / Host B lines from transcript file."""
    if not dialogue_path.exists():
        raise TTSAudioError(f"Dialogue file not found: {dialogue_path}")

    turns: List[DialogueTurn] = []
    with dialogue_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = SPEAKER_RE.match(line)
            if match:
                speaker = "Host A" if "a" in match.group(1).lower() else "Host B"
                text = match.group(2).strip()
                if text:
                    turns.append(DialogueTurn(speaker=speaker, text=text))
            elif turns:
                turns[-1].text = f"{turns[-1].text} {line}".strip()
            else:
                logger.warning("Skipping unparseable line %d before first speaker.", line_no)

    turns = [t for t in turns if t.text]
    if not turns:
        raise TTSAudioError("No parseable dialogue lines found in transcript.")
    logger.info("Parsed %d dialogue turns.", len(turns))
    return turns


def _split_long_block(text: str, max_chars: int) -> List[str]:
    """Split long utterances into API-safe chunks."""
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


def _chunk_text(text: str, max_chars: int = 850) -> List[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    if len(compact) <= max_chars:
        return [compact]
    return [c for c in _split_long_block(compact, max_chars) if c]


def _synthesize_openai_tts_segment(
    *,
    api_key: str,
    tts_model: str,
    voice: str,
    text: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Synthesize one segment with OpenAI TTS and save it to disk."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TTSAudioError("Missing dependency 'openai'. Install requirements.txt.") from exc

    client = OpenAI(api_key=api_key)
    payload = {
        "model": tts_model,
        "voice": voice,
        "input": text,
        "response_format": "mp3",
    }

    logger.debug("Generating TTS segment: %s", output_path.name)
    try:
        with client.audio.speech.with_streaming_response.create(**payload) as response:
            response.stream_to_file(str(output_path))
    except Exception as exc:  # noqa: BLE001
        raise TTSAudioError(f"OpenAI TTS request failed for {output_path.name}: {exc}") from exc


def synthesize_voice_segments(
    *,
    turns: List[DialogueTurn],
    api_keys: Dict[str, str],
    tts_model: str,
    host_a_voice: str,
    host_b_voice: str,
    segments_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Generate per-line/per-chunk audio files and persist them in voice_segments/."""
    api_key = _get_api_key(api_keys, ["OPENAI_API_KEY", "OPENAI"])
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: List[Path] = []
    segment_idx = 1

    for turn_idx, turn in enumerate(turns, start=1):
        chunks = _chunk_text(turn.text)
        logger.info("Turn %d (%s) split into %d chunk(s).", turn_idx, turn.speaker, len(chunks))

        speaker_tag = "HostA" if turn.speaker == "Host A" else "HostB"
        voice = host_a_voice if turn.speaker == "Host A" else host_b_voice

        for chunk in chunks:
            segment_path = segments_dir / f"{segment_idx:03d}_{speaker_tag}.mp3"
            if segment_path.exists() and segment_path.stat().st_size > 0:
                logger.info("Reusing existing segment: %s", segment_path.name)
            else:
                _synthesize_openai_tts_segment(
                    api_key=api_key,
                    tts_model=tts_model,
                    voice=voice,
                    text=chunk,
                    output_path=segment_path,
                    logger=logger,
                )
                logger.info("Saved segment: %s", segment_path.name)
            segment_paths.append(segment_path)
            segment_idx += 1

    if not segment_paths:
        raise TTSAudioError("No voice segments were generated.")
    return segment_paths


def stitch_saved_segments(
    *,
    segments_dir: Path,
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Stitch already saved audio segments from voice_segments/ into one file."""
    if not segments_dir.exists():
        raise TTSAudioError(f"Segments folder not found: {segments_dir}")

    segment_files = sorted(
        [p for p in segments_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".m4a"}],
        key=lambda p: p.name,
    )
    if not segment_files:
        raise TTSAudioError(f"No audio segment files found in: {segments_dir}")

    logger.info("Stitching %d segments from %s", len(segment_files), segments_dir)
    combined = AudioSegment.silent(duration=0)
    for segment_path in segment_files:
        combined += AudioSegment.from_file(segment_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.set_frame_rate(TARGET_SAMPLE_RATE_HZ)
    combined.export(output_path, format="mp3", bitrate=TARGET_MP3_BITRATE)
    logger.info("Final stitched audio saved: %s", output_path)
    return output_path


def synthesize_audio_from_dialogue(
    *,
    dialogue_path: Path,
    api_keys: Dict[str, str],
    tts_model: str,
    host_a_voice: str,
    host_b_voice: str,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """End-to-end TTS pipeline with persisted segments and final stitched audio."""
    turns = parse_dialogue_file(dialogue_path, logger)

    segments_dir = output_dir / "voice_segments"
    synthesize_voice_segments(
        turns=turns,
        api_keys=api_keys,
        tts_model=tts_model,
        host_a_voice=host_a_voice,
        host_b_voice=host_b_voice,
        segments_dir=segments_dir,
        logger=logger,
    )

    final_audio = output_dir / f"{dialogue_path.stem}.mp3"
    return stitch_saved_segments(segments_dir=segments_dir, output_path=final_audio, logger=logger)
