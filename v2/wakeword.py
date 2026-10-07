"""Wake-word gate for DORMANT state. openWakeWord, ONNX runtime, fully offline:
all three .onnx files load from models/openwakeword/ (vendored once via
openwakeword.utils.download_models). No downloads at runtime, no network.

Feeding: 16kHz mono int16 PCM in any chunk size; internally buffered to the
1280-sample (80ms) frames the model expects. Score >= threshold on any frame
fires once, then a refractory window suppresses repeats.

Pretrained phrase is "hey jarvis" (hey_jarvis_v0.1.onnx). A custom "Hey Nova"
needs openWakeWord training (synthetic clips + classifier) — tracked as a
follow-up; the machinery here is phrase-agnostic.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

_MODELS = Path(__file__).resolve().parent.parent / "models" / "openwakeword"
_FRAME = 1280  # samples per predict call (80ms @16k)


class WakeWord:
    def __init__(
        self,
        phrase: str = "hey_jarvis",
        threshold: float = 0.5,
        refractory_s: float = 3.0,
    ):
        self.phrase = phrase
        self.threshold = threshold
        self.refractory_s = refractory_s
        self._model = None
        self._buf = bytearray()
        self._cool_until = 0.0
        self.last_score = 0.0

    @property
    def available(self) -> bool:
        return (_MODELS / "melspectrogram.onnx").exists()

    def _ensure(self):
        if self._model is not None:
            return
        ww = _MODELS / f"{self.phrase}_v0.1.onnx"
        if not ww.exists():
            raise FileNotFoundError(
                f"wake-word model {ww} missing — vendor it once with: "
                "openwakeword.utils.download_models(model_names=['hey_jarvis'], "
                "target_directory='models/openwakeword') and keep the .onnx files"
            )
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[str(ww)],
            melspec_model_path=str(_MODELS / "melspectrogram.onnx"),
            embedding_model_path=str(_MODELS / "embedding_model.onnx"),
            inference_framework="onnx",
        )

    def feed(self, pcm_bytes: bytes) -> bool:
        """Feed a mic chunk. Returns True once per detected phrase."""
        self._ensure()
        now = time.perf_counter()
        if now < self._cool_until:
            self._buf = bytearray()  # don't let stale audio fire on cooldown end
            return False
        self._buf.extend(pcm_bytes)
        fired = False
        while len(self._buf) >= _FRAME * 2:
            frame = bytes(self._buf[: _FRAME * 2])
            del self._buf[: _FRAME * 2]
            scores = self._model.predict(np.frombuffer(frame, dtype=np.int16))
            for name, s in scores.items():
                s = float(s)
                self.last_score = max(self.last_score * 0.9, s)
                if s >= self.threshold:
                    fired = True
            if fired:
                break
        if fired:
            self._cool_until = now + self.refractory_s
            self._buf = bytearray()
            self.last_score = 0.0
        return fired
