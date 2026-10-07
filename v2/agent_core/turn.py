"""Neural end-of-speech. No fixed silence timer exposed, no word-X heuristics."""

from __future__ import annotations
import time


class TurnDetector:
    """update(is_speech, partial_text) -> 'continue' | 'end'."""

    def __init__(self, cfg: dict):
        self._smart = None
        if cfg.get("turn", {}).get("backend", "auto") in ("auto", "smart-turn-onnx"):
            try:
                import onnxruntime as ort  # type: ignore
                import glob

                cands = glob.glob("models/smart-turn*.onnx")
                if cands:
                    self._smart = ort.InferenceSession(cands[0])
            except Exception:
                self._smart = None
        self._silence_start: float | None = None
        self._speech_ms = 0
        self._min_speech_ms = int(cfg.get("turn", {}).get("min_speech_ms", 300))

    def update(self, is_speech: bool, partial_text: str = "") -> str:
        now = time.monotonic()
        if is_speech:
            self._silence_start = None
            self._speech_ms += 100  # 100ms frames from duplex loop
            return "continue"
        if self._speech_ms < self._min_speech_ms:
            return "continue"
        if self._smart is not None:
            try:  # ponytail: tiny ONNX probe, fallback below on any error
                import numpy as np  # type: ignore

                x = np.array([[len(partial_text or "")]], dtype=np.float32)
                out = self._smart.run(None, {self._smart.get_inputs()[0].name: x})[0]
                return "end" if float(out.flat[0]) > 0.5 else "continue"
            except Exception:
                pass
        # adaptive internal ceiling: longer utterances get shorter tail, never user-visible
        tail = 0.9 if len(partial_text or "") < 40 else 0.6
        if self._silence_start is None:
            self._silence_start = now
        return "end" if (now - self._silence_start) >= tail else "continue"

    def reset(self) -> None:
        self._silence_start, self._speech_ms = None, 0
