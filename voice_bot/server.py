from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .llm import chat, clear_history
from .stt import transcribe
from .tts import (
    EDGE_VOICES,
    KOKORO_VOICES,
    synthesize_edge,
    synthesize_kokoro,
)

app = FastAPI(title="Vox")

_static = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((_static / "index.html").read_text())


@app.get("/voices")
async def get_voices() -> JSONResponse:
    return JSONResponse({"kokoro": KOKORO_VOICES, "edge": EDGE_VOICES})


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> JSONResponse:
    """Step 1: Speech-to-text only. Returns transcription immediately."""
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"
    suffix = "." + filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcription = await asyncio.to_thread(transcribe, tmp_path)
        if not transcription:
            return JSONResponse({"error": "No speech detected"}, status_code=400)
        return JSONResponse({"transcription": transcription})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/respond")
async def respond(
    text: str = Form(...),
    tts_engine: str = Form("kokoro"),
    voice: str = Form(""),
    llm_engine: str = Form("ollama"),
) -> JSONResponse:
    """Step 2: LLM + TTS. Takes transcription text, returns response + audio."""
    # 1. LLM
    response_text = await chat(text, engine=llm_engine)

    # 2. TTS
    if tts_engine == "edge":
        selected_voice = voice or "en-IN-NeerjaNeural"
        audio_data, mime = await synthesize_edge(response_text, selected_voice)
    else:
        selected_voice = voice or "af_heart"
        audio_data, mime = await asyncio.to_thread(
            synthesize_kokoro, response_text, selected_voice
        )

    return JSONResponse({
        "response": response_text,
        "audio": base64.b64encode(audio_data).decode(),
        "mime": mime,
    })


@app.post("/clear")
async def clear_chat() -> JSONResponse:
    clear_history()
    return JSONResponse({"ok": True})
