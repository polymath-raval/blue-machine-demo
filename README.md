# Vox

A fully local voice-to-voice chatbot running on your machine. Talk to it, it talks back.

## Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (small model, CPU/int8) |
| LLM | [Qwen3.5 2B](https://ollama.com/library/qwen3.5) via [Ollama](https://ollama.com) |
| Text-to-Speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (local) or [Edge TTS](https://github.com/rany2/edge-tts) (online, Indian/British/Australian accents) |
| Server | FastAPI |
| UI | Vanilla JS, push-to-talk |

## Requirements

- macOS (tested on Apple M4) or Linux
- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.12
- `espeak-ng` (`brew install espeak-ng`) — required by Kokoro TTS
- `ffmpeg` (`brew install ffmpeg`) — for audio format handling

## Setup

```bash
# 1. Pull the LLM
ollama pull qwen3.5:2b

# 2. Install espeak-ng (needed by Kokoro TTS)
brew install espeak-ng

# 3. Install dependencies
uv sync

# 4. Install spaCy language model (needed by Kokoro on first run)
uv pip install pip
uv run python -m spacy download en_core_web_sm
```

## Run

```bash
uv run python main.py
```

Open http://localhost:8000 in your browser.

## Usage

- **Hold Space** or **hold the mic button** to record
- Release to send — the bot transcribes, thinks, and speaks back
- Switch between **Kokoro** (local, no internet) and **Edge TTS** (online, more voices including Indian English)
- Use the **Clear** button to reset the conversation

## Voices

**Kokoro** — fully local, no internet required:
- American English (male & female)
- British English (male & female)

**Edge TTS** — requires internet:
- 🇮🇳 Indian English (Neerja / Prabhat)
- 🇺🇸 American English
- 🇬🇧 British English
- 🇦🇺 Australian English
