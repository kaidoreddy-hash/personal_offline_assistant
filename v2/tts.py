"""Piper TTS for v2 — en + es. Voices load from models/piper/; the first run
on a new machine may fetch a voice once, then it is offline forever.
Windows pip wheels ship a stale compiled-in espeak path, so point Piper at
the real espeak-ng-data dir. Sample rate is resampled to 16k for the sink."""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_SR_OUT = 16000
_MODELS = Path(__file__).resolve().parent.parent / "models"

VOICES: Dict[str, str] = {
    "en": "en_US-ryan-high",
    "es": "es_MX-claude-high",  # absent -> English fallback below
}

_cache: Dict[str, object] = {}


def _espeak_dir():
    try:
        import piper

        c = os.path.join(os.path.dirname(piper.__file__), "espeak-ng-data")
        return c if os.path.isdir(c) else None
    except Exception:
        return None


def _voice_onnx(voice: str) -> Optional[str]:
    p = _MODELS / "piper" / f"{voice}.onnx"
    if p.exists():
        return str(p)
    try:  # one-time fetch
        from piper.download_voices import download_voice

        vdir = _MODELS / "piper"
        vdir.mkdir(parents=True, exist_ok=True)
        download_voice(voice, vdir)
        return str(p) if p.exists() else None
    except Exception:
        return None


def _synth(text: str, voice: str) -> Optional[bytes]:
    try:
        from piper import PiperVoice
    except Exception:
        return None
    onnx = _voice_onnx(voice)
    if onnx is None:
        return None
    try:
        if voice not in _cache:
            kw = {}
            espeak = _espeak_dir()
            if espeak:
                kw["espeak_data_dir"] = espeak
            _cache[voice] = PiperVoice.load(onnx, **kw)
        parts, sr = [], _SR_OUT
        for chunk in _cache[voice].synthesize(text):
            arr = getattr(chunk, "audio_float_array", None)
            if getattr(chunk, "sample_rate", None):
                sr = int(chunk.sample_rate)
            if arr is not None and len(arr):
                parts.append(np.asarray(arr, dtype=np.float32))
        if not parts:
            return None
        audio = (np.concatenate(parts) * 32767.0).astype(np.int16)
        if sr != _SR_OUT:
            ratio = _SR_OUT / float(sr)
            idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
            audio = audio[np.clip(idx, 0, len(audio) - 1)]
        return audio.tobytes()
    except Exception:
        return None


def synthesize(text: str, lang: str = "en") -> bytes:
    text = (text or "").strip()
    if not text:
        return b""
    code = (lang or "en").lower()
    return (
        _synth(text, VOICES.get(code, VOICES["en"]))
        or _synth(text, VOICES["en"])
        or b"\x00\x00" * (_SR_OUT // 4)  # 250ms silence, never choke playback
    )
