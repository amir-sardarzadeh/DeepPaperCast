# DeepPaperCast

Generate long-form, two-host AI podcast episodes from academic PDFs.

The pipeline extracts paper text, writes a strict `Host A` / `Host B` script with an LLM, renders each turn with TTS, and exports a final high-quality MP3.

## Features

- Modular architecture:
  - `main.py` orchestrates the full workflow.
  - `llm_writer.py` handles PDF extraction and LLM script generation.
  - `tts_audio.py` handles dialogue parsing, TTS calls, and audio stitching.
- LLM providers:
  - OpenAI or Anthropic (Claude).
- Voice provider:
  - OpenAI TTS (`gpt-4o-mini-tts`) with configurable host voices.
- Configurable script depth:
  - `DETAIL_LEVEL = "Default"` or `DETAIL_LEVEL = "High"`.
- Output organization:
  - Artifacts move automatically into `Final/<paper_name>/`.
- Logging:
  - Run timeline in `logs/podcast_generator_YYYYMMDD_HHMMSS.log`.

## Project Structure

```text
.
|- main.py
|- llm_writer.py
|- tts_audio.py
|- requirements.txt
|- apit.example.txt
|- apiv.example.txt
`- logs/
```

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in `PATH`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Or with `uv`:

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## API Key Files

Create local key files in the project root:

- `apit.txt` for writer/LLM keys
- `apiv.txt` for voice/TTS keys

Use templates:

- copy `apit.example.txt` to `apit.txt`
- copy `apiv.example.txt` to `apiv.txt`

Example:

```txt
# apit.txt
ANTHROPIC_API_KEY=your_anthropic_key
```

```txt
# apiv.txt
OPENAI_API_KEY=your_openai_key
```

`apit.txt` and `apiv.txt` are git-ignored.

## Quick Configuration

Edit the top of `main.py`:

```python
File = "your_paper.pdf"
Company = "Claude"              # "OpenAI" or "Claude"
Model = "claude-sonnet-4-6"
ExtendedThinking = True         # Anthropic only
ThinkingBudgetTokens = 12000    # Anthropic only
DETAIL_LEVEL = "High"           # "Default" or "High"
```

## Run

```powershell
python main.py
```

Or:

```powershell
uv run --python .\.venv\Scripts\python.exe main.py
```

Optional overrides:

```powershell
python main.py --pdf paper.pdf --company Claude --model claude-sonnet-4-6 --detail-level High
```

## Output

On success, files are placed in:

```text
Final/<paper_name>/
```

Folder contents:

- original PDF
- generated transcript `.txt`
- final podcast `.mp3` (48 kHz, 320 kbps)

## Notes

- Every dialogue line is normalized to `Host A:` or `Host B:`.
- Logs capture API calls, file operations, warnings, and failures.
- Keep real API keys local and never commit secrets.
