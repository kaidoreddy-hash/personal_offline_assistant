"""Silero VAD v5 — ONNX, 512 samples (32ms) per frame @16kHz.
Self-contained: no dependency on v1 core_pipeline. Falls back to RMS/ZCR
proxy if onnxruntime is missing (worse VAD, same API)."""

import math
from pathlib import Path
from typing import Optional

import numpy as np

_MODELS = Path(__file__).resolve().parent.parent / "models"
_CANDIDATES = ("silero_vad.onnx", "silero_vad_v6.onnx", "silero_vad_v5.onnx")


def frame_dbfs(pcm: bytes) -> float:
    """RMS level of a 32ms PCM16 frame, in dBFS."""
    a = np.frombuffer(pcm, dtype=np.int16)
    if len(a) == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    return float(20 * math.log10(rms / 32768.0)) if rms > 1e-9 else -100.0


class SileroVAD:
    """Silero v5 ONNX runner. predict(pcm_bytes) -> speech probability [0,1]."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(
            _MODELS
            / next((n for n in _CANDIDATES if (_MODELS / n).exists()), _CANDIDATES[0])
        )
        self._session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            self._session = None

    def predict(self, pcm_bytes: bytes) -> float:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = (
            audio[:512] if len(audio) >= 512 else np.pad(audio, (0, 512 - len(audio)))
        )
        if self._session is not None:
            try:
                out, self._state = self._session.run(
                    None,
                    {
                        "input": audio.reshape(1, 512),
                        "state": self._state,
                        "sr": self._sr,
                    },
                )
                return float(out[0][0])
            except Exception:
                pass
        # Fallback: RMS/ZCR proxy
        rms = float(np.sqrt(np.mean(audio**2)))
        zcr = int(np.sum(np.diff(audio > 0) != 0))
        return min(1.0, rms * 15.0) if (rms > 0.02 and 15 < zcr < 180) else 0.05

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
