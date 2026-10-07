"""Wake word: openWakeWord (audio, Pi) with transcript fallback (PC)."""

from __future__ import annotations
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent


class WakeWordDetector:
    def __init__(self, cfg: dict):
        self.threshold = float(cfg.get("wakeword_threshold", 0.5))
        self._model: Any = None
        self.backend = "transcript"
        p = cfg.get("wakeword_onnx", "models/hey_jarvis_v0.1.onnx")
        mp = Path(p) if Path(p).is_absolute() else BASE / p
        try:
            import openwakeword  # type: ignore
            from openwakeword.model import Model  # type: ignore

            if mp.exists() and mp.stat().st_size > 0:
                self._model = Model(
                    wakeword_models=[str(mp)], inference_framework="onnx"
                )
                self.backend = "openwakeword"
        except Exception:
            # transcript fallback: duplex.py matches wakeword inside STT text
            self._model = None

    def predict(self, pcm16: bytes) -> bool:
        if self._model is None or not pcm16:
            return False
        try:
            import numpy as np  # type: ignore

            for _, score in self._model.predict(
                np.frombuffer(pcm16, dtype=np.int16)
            ).items():
                if score >= self.threshold:
                    self.reset()
                    return True
        except Exception:
            pass
        return False

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
