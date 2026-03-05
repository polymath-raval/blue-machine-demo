from __future__ import annotations

import io

import edge_tts
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# ── Kokoro ────────────────────────────────────────────────────────────────────

_pipeline: KPipeline | None = None
KOKORO_SAMPLE_RATE = 24000

KOKORO_VOICES: dict[str, str] = {
    "af_heart":   "🇺🇸 American Female – Heart",
    "af_bella":   "🇺🇸 American Female – Bella",
    "af_sarah":   "🇺🇸 American Female – Sarah",
    "af_nicole":  "🇺🇸 American Female – Nicole",
    "am_michael": "🇺🇸 American Male – Michael",
    "am_adam":    "🇺🇸 American Male – Adam",
    "bf_emma":    "🇬🇧 British Female – Emma",
    "bf_alice":   "🇬🇧 British Female – Alice",
    "bm_george":  "🇬🇧 British Male – George",
    "bm_daniel":  "🇬🇧 British Male – Daniel",
}


def get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a")
    return _pipeline


def synthesize_kokoro(text: str, voice: str = "af_heart") -> tuple[bytes, str]:
    pipeline = get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=1.0)]
    audio_array = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio_array, KOKORO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


# ── Edge TTS ──────────────────────────────────────────────────────────────────

EDGE_VOICES: dict[str, str] = {
    "en-IN-NeerjaNeural":  "🇮🇳 Indian English Female – Neerja",
    "en-IN-PrabhatNeural": "🇮🇳 Indian English Male – Prabhat",
    "en-US-JennyNeural":   "🇺🇸 American English Female – Jenny",
    "en-US-GuyNeural":     "🇺🇸 American English Male – Guy",
    "en-GB-SoniaNeural":   "🇬🇧 British English Female – Sonia",
    "en-GB-RyanNeural":    "🇬🇧 British English Male – Ryan",
    "en-AU-NatashaNeural": "🇦🇺 Australian English Female – Natasha",
    "en-AU-WilliamNeural": "🇦🇺 Australian English Male – William",
}


async def synthesize_edge(text: str, voice: str = "en-IN-NeerjaNeural") -> tuple[bytes, str]:
    communicate = edge_tts.Communicate(text, voice)
    mp3_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])
    return b"".join(mp3_chunks), "audio/mpeg"
