#!/usr/bin/env python3
"""Standalone utility to stitch existing voice_segments into one audio file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from pydub import AudioSegment

TARGET_SAMPLE_RATE_HZ = 48_000
TARGET_MP3_BITRATE = "320k"
# Quick-start config (edit then run: python stitch_audio.py)
SEGMENTS_DIR = r"Final\Your_Paper_Title\voice_segments"
OUTPUT_FILE = ""


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
    """Stitch segment files into one final MP3."""
    segment_files = _list_segment_files(segments_dir)
    logger.info("Found %d segment files.", len(segment_files))

    combined = AudioSegment.silent(duration=0)
    for segment_path in segment_files:
        logger.info("Adding segment: %s", segment_path.name)
        combined += AudioSegment.from_file(segment_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.set_frame_rate(TARGET_SAMPLE_RATE_HZ)
    combined.export(output_path, format="mp3", bitrate=TARGET_MP3_BITRATE)
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
