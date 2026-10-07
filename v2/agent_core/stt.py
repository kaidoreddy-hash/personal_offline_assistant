"""STT: faster-whisper if installed (offline path), else mock passthrough."""

from __future__ import annotations
from pathlib import Path


class STT:
    def __init__(self, cfg: dict):
        s = cfg.get("stt", {})
        self.backend = "mock"
        self._model = None
        if s.get("backend", "auto") in ("auto", "faster-whisper"):
            try:
                from faster_whisper import WhisperModel  # type: ignore

                mp = s.get("model_path", "models/stt-tiny.en")
                if Path(mp).exists():
                    self._model = WhisperModel(mp, device="cpu", compute_type="int8")
                    self.backend = "faster-whisper"
            except Exception:
                self._model = None

    def transcribe(self, pcm16: bytes) -> str:
        if self._model is None:
            return ""  # mock: text arrives via WS partial/final events on PC
        import numpy as np, tempfile, wave  # type: ignore

        a = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((a * 32767).astype("<i2").tobytes())
            segs, _ = self._model.transcribe(f.name)
            return " ".join(s.text for s in segs).strip()
