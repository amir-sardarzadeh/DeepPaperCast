# DeepPaperCast
<<<<<<< HEAD

Turn academic PDFs into a two-host AI podcast with detailed technical explanations, then export high-quality MP3 audio.

## GitHub Repository Name

Recommended repo name:

`deeppapercast`

## GitHub Description (Short)

Generate detailed two-host AI podcast episodes from research PDFs using Claude/OpenAI LLMs and OpenAI/ElevenLabs TTS, exported as 48kHz 320kbps MP3.

## What This Project Does

1. Reads a research paper PDF.
2. Generates a detailed, structured Host A / Host B dialogue using an LLM.
3. Synthesizes each turn with TTS (distinct voices per host).
4. Concatenates segments with ffmpeg.
5. Exports final audio as MP3 at `48,000 Hz` and `320 kbps`.
6. Organizes PDF + transcript + MP3 into a folder named after the paper.

## Architecture

- `main.py`: Orchestrator, config, key-file loading, run pipeline, output organization.
- `llm_writer.py`: PDF extraction + LLM dialogue generation.
- `tts_audio.py`: Dialogue parsing + TTS calls + ffmpeg stitching/export.

## Requirements

- Python 3.10+ (recommended: project-local environment via `uv`)
- ffmpeg available in PATH
- API keys in local files:
  - `apit.txt` for writer/LLM
  - `apiv.txt` for voice/TTS

## Setup (Recommended with uv)

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## API Key Files

Use the provided templates:

- `apit.example.txt` -> copy to `apit.txt`
- `apiv.example.txt` -> copy to `apiv.txt`

`apit.txt` (writer/LLM):

```txt
ANTHROPIC_API_KEY=your_anthropic_key
```

`apiv.txt` (voice/TTS):

```txt
OPENAI_API_KEY=your_openai_key
```

Important:

- Do not commit `apit.txt` or `apiv.txt` (they are ignored by `.gitignore`).
- Keep your real keys local only.

## Quick Configuration

Edit the top of `main.py`:

```python
File = "your_paper.pdf"
Company = "Claude"      # or "OpenAI"
Model = "claude-sonnet-4-6"
ExtendedThinking = True
ThinkingBudgetTokens = 12000
```

## Run

```powershell
uv run --python .\.venv\Scripts\python.exe main.py
```

## Output

Generated files are moved into:

`outputs/<paper-title>/`

- original PDF
- transcript `.txt`
- podcast `.mp3` (48kHz, 320kbps)

## Notes

- This project is set to detailed-only dialogue generation (not short summaries).
- Anthropic extended thinking is supported and configurable.
- For Anthropic thinking mode, temperature adjustments are automatically handled for compatibility.
=======
Generate detailed two-host AI podcast episodes from research PDFs using online LLMs and TTS, like how NotebookLM works.
>>>>>>> bd965d4860fa0c434d0277d2e13b5077270ec0bb
