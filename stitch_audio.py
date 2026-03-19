#!/usr/bin/env python3
"""Standalone utility to stitch existing voice_segments into one audio file."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

TARGET_SAMPLE_RATE_HZ = 48_000
TARGET_MP3_BITRATE = "320k"
# Quick-start config (edit then run: python stitch_audio.py)
SEGMENTS_DIR = r"Final\Your_Paper_Title\voice_segments"
OUTPUT_FILE = ""


def _ffmpeg_concat_entry(path: Path) -> str:
    """Build one safe concat-demuxer manifest line for ffmpeg."""
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


class StitchAudioError(RuntimeError):
    """Raised when standalone segment stitching fails."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stitch persisted TTS segments into one podcast file.")
    parser.add_argument("--segments-dir", default=SEGMENTS_DIR, help="Path to voice_segments folder")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output mp3 path (default: <segments_dir>/stitched_podcast.mp3)")
    return parser.parse_args()


def build_logger() -> logging.Logger:
    """Create logger for standalone stitching runs."""
    logger = logging.getLogger("stitch_audio")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _list_segment_files(segments_dir: Path) -> List[Path]:
    """List supported audio files in stable alphabetical order."""
    if not segments_dir.exists():
        raise StitchAudioError(f"Segments folder not found: {segments_dir}")
    if not segments_dir.is_dir():
        raise StitchAudioError(f"Provided path is not a directory: {segments_dir}")

    files = sorted(
        [p for p in segments_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".m4a"}],
        key=lambda p: p.name,
    )
    if not files:
        raise StitchAudioError(f"No MP3/M4A files found in: {segments_dir}")
    return files


def stitch_segments(*, segments_dir: Path, output_path: Path, logger: logging.Logger) -> Path:
    """Stitch segment files into one final MP3 using ffmpeg."""
    segment_files = _list_segment_files(segments_dir)
    logger.info("Found %d segment files.", len(segment_files))

    manifest_path = segments_dir / "_ffmpeg_concat_input.txt"
    manifest_lines = [_ffmpeg_concat_entry(p) for p in segment_files]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-ar",
        str(TARGET_SAMPLE_RATE_HZ),
        "-b:a",
        TARGET_MP3_BITRATE,
        "-vn",
        str(output_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise StitchAudioError("ffmpeg not found in PATH. Install ffmpeg and try again.") from exc
    finally:
        try:
            manifest_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-1200:]
        raise StitchAudioError(f"ffmpeg stitching failed: {stderr_tail or 'Unknown ffmpeg error.'}")

    logger.info("Exported stitched audio: %s", output_path)
    return output_path


def run() -> int:
    """Run the standalone stitch flow."""
    args = parse_args()
    logger = build_logger()

    try:
        segments_dir = Path(args.segments_dir).expanduser().resolve()
        output = (
            Path(args.output).expanduser().resolve()
            if args.output and str(args.output).strip()
            else (segments_dir / "stitched_podcast.mp3").resolve()
        )
        stitch_segments(segments_dir=segments_dir, output_path=output, logger=logger)
        logger.info("Done. Output: %s", output)
        return 0
    except StitchAudioError as exc:
        logger.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
