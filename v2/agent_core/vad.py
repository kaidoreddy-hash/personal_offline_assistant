"""VAD: Silero-v5 ONNX via onnxruntime (Pi-friendly, ~2MB), else energy fallback."""

from __future__ import annotations
from pathlib import Path
import struct


def _vad_onnx_path(cfg: dict) -> Path:
    p = cfg.get("vad", {}).get("onnx_path", "models/silero_vad_v5.onnx")
    base = Path(__file__).resolve().parent.parent
    return Path(p) if Path(p).is_absolute() else base / p


class VAD:
    def __init__(self, cfg: dict):
        v = cfg.get("vad", {})
        self.threshold = float(v.get("threshold", 0.6))
        self._sess = None
        # ponytail: onnxruntime only (~50MB RAM). torch.hub path deleted —
        # it pulled ~800MB torch + downloaded at runtime, kills Pi 4GB.
        if v.get("backend", "auto") in ("auto", "silero-v5"):
            mp = _vad_onnx_path(cfg)
            if mp.exists():
                try:
                    import onnxruntime as ort  # type: ignore

                    self._sess = ort.InferenceSession(str(mp))
                except Exception:
                    self._sess = None

    def is_speech(self, pcm16: bytes) -> bool:
        if self._sess is not None:
            try:
                import numpy as np  # type: ignore

                a = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
                # silero-v5 onnx: [batch, samples] float32 -> prob
                inp = self._sess.get_inputs()[0].name
                p = float(self._sess.run(None, {inp: a[None, :]})[0].flat[0])
                return p >= self.threshold
            except Exception:
                pass
        # energy fallback (internal only, never shown as "silence timer")
        if not pcm16:
            return False
        n = len(pcm16) // 2
        s = struct.unpack(f"<{n}h", pcm16)
        rms = (sum(x * x for x in s) / max(n, 1)) ** 0.5 / 32768.0
        return rms > 0.02
