"""Piper TTS for v2 — English-only scope. Voice loads from models/piper/;
the first run on a new machine may fetch it once, then it is offline
forever. Windows pip wheels ship a stale compiled-in espeak path, so point
Piper at the real espeak-ng-data dir. Output resampled to 16k for the sink."""

import os
from pathlib import Path
from typing import Optional

import numpy as np

_SR_OUT = 16000
_MODELS = Path(__file__).resolve().parent.parent / "models"

VOICE = "en_US-ryan-high"

_cache: dict = {}


def _espeak_dir():
    try:
        import piper

        c = os.path.join(os.path.dirname(piper.__file__), "espeak-ng-data")
        return c if os.path.isdir(c) else None
    except Exception:
        return None


def _voice_onnx() -> Optional[str]:
    p = _MODELS / "piper" / f"{VOICE}.onnx"
    if p.exists():
        return str(p)
    try:  # one-time fetch
        from piper.download_voices import download_voice

        vdir = _MODELS / "piper"
        vdir.mkdir(parents=True, exist_ok=True)
        download_voice(VOICE, vdir)
        return str(p) if p.exists() else None
    except Exception:
        return None


def _synth(text: str) -> Optional[bytes]:
    try:
        from piper import PiperVoice
    except Exception:
        return None
    onnx = _voice_onnx()
    if onnx is None:
        return None
    try:
        if VOICE not in _cache:
            kw = {}
            espeak = _espeak_dir()
            if espeak:
                kw["espeak_data_dir"] = espeak
            _cache[VOICE] = PiperVoice.load(onnx, **kw)
        parts, sr = [], _SR_OUT
        for chunk in _cache[VOICE].synthesize(text):
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
    return _synth(text) or b"\x00\x00" * (_SR_OUT // 4)  # never choke playback
