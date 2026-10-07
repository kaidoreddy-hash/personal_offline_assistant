"""SmartTurn v3 local judge — ONNX. 16kHz mono PCM16, last 8s tail.
Audio-native, so it works for es + en alike. predict() -> None when onnxruntime
or the model is missing; callers then fall back to the silence ceiling.

Log-mel front-end is whisper_features.py (numpy-only). transformers would work
but costs 4.9s import + 199MB RSS and drags torch in — unacceptable on a 4GB
Pi where the LLM needs the headroom."""

from pathlib import Path
from typing import Optional

import numpy as np

from v2.whisper_features import compute_whisper_log_mel_features

_MODELS = Path(__file__).resolve().parent.parent / "models"
_CANDIDATES = (
    "smart-turn-v3.2-cpu.onnx",
    "smart-turn-v3.1-cpu.onnx",
    "smart-turn-v3.0.onnx",
)
_MAX_S = 8.0  # model's hard input window
_SR = 16000


class SmartTurnJudge:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(
            _MODELS
            / next((n for n in _CANDIDATES if (_MODELS / n).exists()), _CANDIDATES[0])
        )
        self._session = None
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            self._session = None

    def available(self) -> bool:
        return self._session is not None

    def predict(self, pcm_bytes: bytes) -> Optional[float]:
        """P(complete) in [0,1], or None if the model isn't present."""
        if self._session is None:
            return None
        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            feats = compute_whisper_log_mel_features(audio, do_normalize=True)
            out = self._session.run(None, {"input_features": feats[np.newaxis, ...]})
            return float(np.asarray(out[0]).ravel()[0])
        except Exception:
            return None
