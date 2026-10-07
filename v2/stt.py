"""Whistle STT adapter — en + es, offline. The cactus-needle wheel ships the
aarch64 engine and a 16.9 MB .cact, both cached after the first run.
Batch mode is the fast path (measured RTF ~0.02 under 5s; cliff at ~6s, so
turns must stay short — TurnSegmenter caps the buffer)."""

import os
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("NEEDLE_TELEMETRY", "0")

_shared = None


def _w():
    global _shared
    if _shared is None:
        import needle  # cactus-needle

        _shared = needle.Whistle()
    return _shared


def transcribe(
    pcm16_bytes: bytes, language: Optional[str] = None
) -> Tuple[str, str, float]:
    """pcm16 @16k mono -> (text, lang_code, confidence). ('' , '', 0.0) on
    anything the engine can't hear."""
    audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < int(16000 * 0.25):
        return "", "", 0.0
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    if 20 * float(np.log10(rms)) < -45.0:
        return "", "", 0.0  # room tone: never spend model time on it
    try:
        r = _w().transcribe(audio, language=language)
        text = (r.get("text") or "").strip()
        # No per-utterance ASR confidence: 1.0 when text exists. Whistle
        # returns empty (not invented text) on silence and steady noise.
        return text, (r.get("language") or "").lower() or "en", 1.0 if text else 0.0
    except Exception:
        return "", "", 0.0
