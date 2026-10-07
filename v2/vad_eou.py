"""Silero VAD runner — ONNX, self-contained copy so v2 tests standalone.
Frame: 512 samples @16k = 32ms. Without onnxruntime, falls back to an
RMS/ZCR proxy so the loop still runs (worse VAD, same shape)."""

from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent.parent / "models"
_CANDIDATES = ("silero_vad.onnx", "silero_vad_v6.onnx", "silero_vad_v5.onnx")


class SileroVAD:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or next(
            str(_MODELS / n) for n in _CANDIDATES if (_MODELS / n).exists()
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
        # ponytail: RMS/ZCR proxy, not a real VAD. Install onnxruntime.
        rms = float(np.sqrt(np.mean(audio**2)))
        zcr = int(np.sum(np.diff(audio > 0) != 0))
        return min(1.0, rms * 15.0) if (rms > 0.02 and 15 < zcr < 180) else 0.05

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
