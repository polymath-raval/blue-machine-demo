from __future__ import annotations

from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        # small model, CPU with INT8 quantization — fast enough on M4
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> str:
    model = get_model()
    segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
    return " ".join(seg.text for seg in segments).strip()
