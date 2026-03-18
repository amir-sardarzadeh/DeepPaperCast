"""TTS audio generation and stitching module for podcast dialogue files."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SPEAKER_RE = re.compile(r"^(Host\s*A|Host\s*B|Person\s*A|Person\s*B)\s*:\s*(.*)$", re.IGNORECASE)
TARGET_SAMPLE_RATE_HZ = 48_000
TARGET_MP3_BITRATE = "320k"

VOICE_STYLE_GUIDE = (
    "Voice affect: bright, engaged, intellectually curious, with genuine excitement about new insights. "
    "Tone: upbeat, conversational, relatable, like a friendly knowledgeable peer. "
    "Pacing: dynamic and natural; brisk during back-and-forth banter, then deliberately slower on complex ideas and core takeaways. "
    "Emotion: warm, enthusiastic, approachable, with a subtle smile and occasional 'aha' energy. "
    "Emphasis: lean into transition phrases such as 'But here's the crazy part' and 'Think about it like this', and make analogies pop. "
    "Pronunciation: crisp and articulate with modern podcast cadence, not formal news-anchor delivery. "
    "Inflection: use conversational rises on clarifying questions. "
    "Pauses: add brief natural beats, especially before punchy analogies and right after dense technical terms."
)


class TTSAudioError(RuntimeError):
    """Raised when dialogue parsing or TTS synthesis fails."""


@dataclass
class DialogueTurn:
    """One parsed speaker turn from the dialogue text file."""

    speaker: str
    text: str


class BaseTTSClient:
    """Base interface for provider-specific TTS clients."""

    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        """Synthesize text into MP3 bytes."""
        raise NotImplementedError


class OpenAITTSClient(BaseTTSClient):
    """OpenAI text-to-speech client wrapper."""

    def __init__(self, *, model: str, api_key: str, logger: logging.Logger) -> None:
        """Initialize OpenAI TTS client."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TTSAudioError("Missing dependency 'openai'. Install requirements.txt.") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logger

    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        """Generate MP3 bytes from OpenAI TTS."""
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "instructions": _voice_instruction(speaker),
            "response_format": "mp3",
        }
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            self.logger.debug("OpenAI TTS request. model=%s voice=%s speaker=%s", self.model, voice, speaker)
            with self.client.audio.speech.with_streaming_response.create(**payload) as response:
                response.stream_to_file(str(temp_path))
        except Exception as exc:  # noqa: BLE001
            if "instruction" in str(exc).lower():
                self.logger.warning("OpenAI TTS model rejected instructions; retrying without instructions.")
                payload.pop("instructions", None)
                with self.client.audio.speech.with_streaming_response.create(**payload) as response:
                    response.stream_to_file(str(temp_path))
            else:
                temp_path.unlink(missing_ok=True)
                raise TTSAudioError(f"OpenAI TTS request failed: {exc}") from exc

        try:
            return temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)


class ElevenLabsTTSClient(BaseTTSClient):
    """ElevenLabs text-to-speech client wrapper."""

    def __init__(self, *, model: str, api_key: str, logger: logging.Logger) -> None:
        """Initialize ElevenLabs TTS client."""
        self.model = model
        self.api_key = api_key
        self.logger = logger

    def synthesize(self, *, text: str, voice: str, speaker: str) -> bytes:
        """Generate MP3 bytes from ElevenLabs TTS."""
        try:
            import requests
        except ImportError as exc:
            raise TTSAudioError("Missing dependency 'requests'. Install requirements.txt.") from exc

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
        }
        self.logger.debug("ElevenLabs TTS request. model=%s voice=%s speaker=%s", self.model, voice, speaker)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            raise TTSAudioError(f"ElevenLabs TTS request failed (possible rate limit or network issue): {exc}") from exc


def _voice_instruction(speaker: str) -> str:
    """Return style instruction for the speaker voice."""
    if speaker == "Host A":
        role_style = (
            "You are Host A, the steady guide: confident, clear, and structured, while still playful and curious."
        )
    else:
        role_style = (
            "You are Host B, the curious co-host: energetic, warm, and quick to ask smart clarifying questions."
        )
    return f"{role_style} {VOICE_STYLE_GUIDE}"


def _get_api_key(api_keys: Dict[str, str], aliases: List[str]) -> str:
    """Resolve an API key by trying several alias names."""
    for alias in aliases:
        value = api_keys.get(alias.upper(), "").strip()
        if value:
            return value
    raise TTSAudioError(f"Missing API key. Expected one of: {', '.join(aliases)}")


def _build_tts_client(
    *,
    provider: str,
    model: str,
    api_keys: Dict[str, str],
    logger: logging.Logger,
) -> BaseTTSClient:
    """Build the selected provider TTS client."""
    p = provider.lower()
    if p == "openai":
        return OpenAITTSClient(
            model=model,
            api_key=_get_api_key(api_keys, ["OPENAI_API_KEY", "OPENAI"]),
            logger=logger,
        )
    if p == "elevenlabs":
        return ElevenLabsTTSClient(
            model=model,
            api_key=_get_api_key(api_keys, ["ELEVENLABS_API_KEY", "ELEVENLABS"]),
            logger=logger,
        )
    raise TTSAudioError(f"Unsupported TTS provider: {provider}")


def _split_long_block(text: str, max_chars: int) -> List[str]:
    """Split text into chunks suitable for TTS API input length constraints."""
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
            bucket_len = 0
            for word in words:
                add = len(word) + (1 if bucket else 0)
                if bucket_len + add > max_chars and bucket:
                    chunks.append(" ".join(bucket))
                    bucket = [word]
                    bucket_len = len(word)
                else:
                    bucket.append(word)
                    bucket_len += add
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


def _split_for_tts(text: str, max_chars: int = 850) -> List[str]:
    """Normalize and split text into API-safe chunks for TTS synthesis."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    if len(compact) <= max_chars:
        return [compact]
    return [chunk for chunk in _split_long_block(compact, max_chars) if chunk]


def parse_dialogue_file(dialogue_path: Path, logger: logging.Logger) -> List[DialogueTurn]:
    """Parse a dialogue text file into speaker turns for Host/Person A and B."""
    if not dialogue_path.exists():
        raise TTSAudioError(f"Dialogue file not found: {dialogue_path}")

    logger.info("Parsing dialogue file: %s", dialogue_path)
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
                else:
                    logger.warning("Empty dialogue content on line %d.", line_no)
            elif turns:
                turns[-1].text = f"{turns[-1].text} {line}".strip()
            else:
                logger.warning("Ignoring unparseable line %d before first speaker: %s", line_no, line)

    turns = [turn for turn in turns if turn.text]
    if not turns:
        raise TTSAudioError("No parseable dialogue lines found. Expected 'Host A:'/'Host B:' format.")
    logger.info("Parsed dialogue turns: %d", len(turns))
    return turns


def _stitch_audio(
    *,
    turns: Iterable[DialogueTurn],
    tts_client: BaseTTSClient,
    host_a_voice: str,
    host_b_voice: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Synthesize each turn and stitch all clips into a single audio file."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise TTSAudioError("ffmpeg not found in PATH. Install ffmpeg and retry.")
    logger.info("Starting TTS synthesis and stitching.")
    with tempfile.TemporaryDirectory(prefix="podcast_segments_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        segment_paths: List[Path] = []
        segment_index = 0

        for turn_index, turn in enumerate(turns, start=1):
            voice = host_a_voice if turn.speaker == "Host A" else host_b_voice
            chunks = _split_for_tts(turn.text, max_chars=850)
            logger.info("Turn %d (%s): %d text chunk(s).", turn_index, turn.speaker, len(chunks))

            for chunk_index, chunk in enumerate(chunks, start=1):
                logger.debug("Synthesizing turn %d chunk %d.", turn_index, chunk_index)
                audio_bytes = tts_client.synthesize(text=chunk, voice=voice, speaker=turn.speaker)
                segment_path = temp_dir / f"seg_{segment_index:05d}.mp3"
                segment_path.write_bytes(audio_bytes)
                segment_paths.append(segment_path)
                segment_index += 1

        if not segment_paths:
            raise TTSAudioError("No audio segments were generated from dialogue.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _concat_with_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            segment_paths=segment_paths,
            output_path=output_path,
            logger=logger,
        )
    logger.info("Audio export complete. Segments synthesized: %d", len(segment_paths))


def _concat_with_ffmpeg(
    *,
    ffmpeg_bin: str,
    segment_paths: List[Path],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Concatenate MP3 segments with ffmpeg and export final MP3 at 48 kHz / 320 kbps."""
    with tempfile.TemporaryDirectory(prefix="podcast_concat_") as concat_dir_str:
        concat_dir = Path(concat_dir_str)
        list_file = concat_dir / "concat.txt"
        temp_mp3 = concat_dir / "combined.mp3"

        # Concat demuxer line format: file '<absolute_path>'
        # Escape single quotes to keep ffmpeg parser safe.
        lines = []
        for path in segment_paths:
            escaped = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger.info("Concatenating %d audio segments with ffmpeg.", len(segment_paths))
        concat_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(temp_mp3),
        ]
        concat_copy = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)

        # Fallback to re-encode in case stream-copy concat fails.
        if concat_copy.returncode != 0:
            logger.warning("ffmpeg stream-copy concat failed; retrying with re-encode.")
            concat_reencode_cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(temp_mp3),
            ]
            concat_reencode = subprocess.run(
                concat_reencode_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if concat_reencode.returncode != 0:
                raise TTSAudioError(
                    f"ffmpeg concat failed. stderr={concat_reencode.stderr.strip() or concat_copy.stderr.strip()}"
                )

        export_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temp_mp3),
            "-c:a",
            "libmp3lame",
            "-ar",
            str(TARGET_SAMPLE_RATE_HZ),
            "-b:a",
            TARGET_MP3_BITRATE,
            str(output_path),
        ]
        export = subprocess.run(export_cmd, capture_output=True, text=True, check=False)
        if export.returncode != 0:
            raise TTSAudioError(f"ffmpeg mp3 export failed. stderr={export.stderr.strip()}")


def synthesize_audio_from_dialogue(
    *,
    dialogue_path: Path,
    api_keys: Dict[str, str],
    tts_provider: str,
    tts_model: str,
    host_a_voice: str,
    host_b_voice: str,
    output_format: str,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Parse dialogue file, generate speaker audio, and save final podcast file."""
    turns = parse_dialogue_file(dialogue_path, logger)
    tts_client = _build_tts_client(provider=tts_provider, model=tts_model, api_keys=api_keys, logger=logger)
    logger.info("TTS client ready. Provider=%s, Model=%s", tts_provider, tts_model)

    paper_name = dialogue_path.stem
    if output_format.lower() != "mp3":
        logger.warning("Overriding requested output format '%s' to 'mp3' (48 kHz, 320 kbps).", output_format)
    output_path = output_dir / f"{paper_name}.mp3"
    _stitch_audio(
        turns=turns,
        tts_client=tts_client,
        host_a_voice=host_a_voice,
        host_b_voice=host_b_voice,
        output_path=output_path,
        logger=logger,
    )
    return output_path
