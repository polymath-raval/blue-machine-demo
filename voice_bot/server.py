from __future__ import annotations

import asyncio
import base64
import json
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .llm import chat, clear_history, stream_chat
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


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split buffer on sentence-ending punctuation followed by whitespace.
    Returns (complete_sentences, remaining_buffer).
    Only returns sentences of at least 6 chars to avoid TTS on tiny fragments.
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    sentences = [p.strip() for p in parts[:-1] if len(p.strip()) >= 6]
    return sentences, parts[-1]


async def _tts(text: str, tts_engine: str, voice: str) -> tuple[bytes, str]:
    if tts_engine == "edge":
        return await synthesize_edge(text, voice or "en-IN-NeerjaNeural")
    return await asyncio.to_thread(synthesize_kokoro, text, voice or "af_heart")


# ── Streaming respond endpoint ────────────────────────────────────────────────

@app.post("/respond")
async def respond(
    text: str = Form(...),
    tts_engine: str = Form("kokoro"),
    voice: str = Form(""),
    llm_engine: str = Form("ollama"),
) -> StreamingResponse:
    """
    SSE stream: LLM tokens → sentence buffer → TTS per sentence → audio chunks.
    Events:
      {"type": "text",  "chunk": "..."}          — sentence text
      {"type": "audio", "data": "<b64>", "mime": "..."}  — audio for that sentence
      {"type": "error", "message": "..."}
      {"type": "done"}
    """
    async def generate():
        buffer = ""
        try:
            async for token in stream_chat(text, engine=llm_engine):
                buffer += token
                sentences, buffer = _extract_sentences(buffer)

                for sentence in sentences:
                    yield f"data: {json.dumps({'type': 'text', 'chunk': sentence})}\n\n"
                    try:
                        audio_data, mime = await _tts(sentence, tts_engine, voice)
                        yield f"data: {json.dumps({'type': 'audio', 'data': base64.b64encode(audio_data).decode(), 'mime': mime})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            # Flush remaining buffer (last sentence without trailing punctuation)
            remainder = buffer.strip()
            if remainder:
                yield f"data: {json.dumps({'type': 'text', 'chunk': remainder})}\n\n"
                try:
                    audio_data, mime = await _tts(remainder, tts_engine, voice)
                    yield f"data: {json.dumps({'type': 'audio', 'data': base64.b64encode(audio_data).decode(), 'mime': mime})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/clear")
async def clear_chat() -> JSONResponse:
    clear_history()
    return JSONResponse({"ok": True})
