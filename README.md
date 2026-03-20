# DeepPaperCast

Turn research PDFs into:
- a two-host AI podcast transcript
- a stitched podcast audio file
- an optional LaTeX teaching document and compiled PDF

All outputs are organized under:

```text
Final/<paper_name>/
```

## Architecture

- `main.py`: full pipeline (PDF -> transcript -> TTS segments -> stitched audio)
- `llm_writer.py`: shared writer module for main flow (PDF extraction + dialogue generation)
- `tts_audio.py`: TTS generation, persistent `voice_segments/`, and stitching
- `script_only.py`: standalone transcript-only pipeline (no TTS, no dependency on `llm_writer.py`)
- `latex.py`: standalone LaTeX explainer pipeline (`.tex` + compiled PDF)
- `stitch_audio.py`: standalone segment stitcher (rebuild final MP3 from `voice_segments/`)

## Key Output Behavior

- Each run creates/uses `Final/<paper_name>/`
- `main.py` and `script_only.py` save transcript as `<paper_name>.txt`
- `tts_audio.py` saves chunked audio in:
  - `Final/<paper_name>/voice_segments/`
- `latex.py` saves:
  - original input PDF copy
  - `<paper_name>_explained.tex`
  - `<paper_name>_explained.pdf`
- `processing.log` is saved inside each paper folder

## Requirements

- Python 3.10+
- `ffmpeg` in PATH (required by audio generation/stitching)
- TeX distribution with `pdflatex` in PATH (required by `latex.py`)

Install Python packages:

```powershell
python -m pip install -r requirements.txt
```

## API Keys (`apit.txt` + `apiv.txt`)

Create these files in project root:

```txt
# apit.txt (text generation)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key_if_using_openai_writer
```

```txt
# apiv.txt (voice generation)
OPENAI_API_KEY=your_openai_key
```

Notes:
- `main.py` uses:
  - `apit.txt` for writer (`llm_writer.py`)
  - `apiv.txt` for voice (`tts_audio.py`)
- `script_only.py` and `latex.py` are text-only and default to `apit.txt`.

## Quick Start

### 1. Full pipeline

Edit config at top of `main.py` (or pass CLI args), then run:

```powershell
python main.py
```

`DETAIL_LEVEL` options in `main.py`:
- `Default`: concise podcast script
- `Medium`: previous single-pass high-detail behavior
- `High`: 3-part macro-chunk deep dive
  - Part 1: background/problem framing
  - Part 2: methodology + formulas
  - Part 3: results, limitations, implications, outro

### 2. Transcript only

Edit config at top of `script_only.py` (or pass `--pdf`), then run:

```powershell
python script_only.py
```

### 3. LaTeX explainer

Edit config at top of `latex.py` (or pass `--pdf`), then run:

```powershell
python latex.py
```

### 4. Re-stitch existing segments

Edit config at top of `stitch_audio.py` or pass path:

```powershell
python stitch_audio.py --segments-dir "Final/<paper_name>/voice_segments"
```

Default stitch output is saved in the same segments folder as `stitched_podcast.mp3`.

## TeX Installation (for `latex.py`)

- Windows: install MiKTeX and ensure `pdflatex` is in PATH
- macOS: install MacTeX (`mactex-no-gui` is fine)
- Ubuntu/Debian:

```bash
sudo apt update
sudo apt install texlive-latex-extra texlive-fonts-recommended
```

Verify:

```powershell
pdflatex --version
```

## Notes

- Large token budgets can be slow and expensive.
- Anthropic large requests automatically use streaming in this codebase.
- Anthropic token budgets are auto-clamped to model limits and valid thinking/output relationships.
- Audio stitching uses `ffmpeg` directly (no `pydub` dependency), which avoids Python 3.14 `audioop` issues.
- Keep real keys local; do not commit `apit.txt`, `apiv.txt`, or `api.txt`.
