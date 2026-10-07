"""VAD: Silero-v5 ONNX via onnxruntime (Pi-friendly, ~2MB), with energy fallback.

Fixes the silent failure: Silero v5 takes THREE inputs (input, state
[2,B,128], sr). Passing only `input` raised on every frame and dropped us to
the crude RMS gate, which misses quiet speech. Now the recurrent state is
carried between frames.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent


def _vad_onnx_path(cfg: dict) -> Path:
    p = cfg.get("vad", {}).get("onnx_path", "models/silero_vad_v5.onnx")
    return Path(p) if Path(p).is_absolute() else BASE / p


class VAD:
    """Stateful Silero-v5 ONNX VAD. 96ms frames -> speech probability."""

    def __init__(self, cfg: dict):
        v = cfg.get("vad", {})
        self.threshold = float(v.get("threshold", 0.2))
        self._sess: Any = None
        self._state = None
        self._names: list[str] = []
        if v.get("backend", "auto") in ("auto", "silero-v5"):
            mp = _vad_onnx_path(cfg)
            if mp.exists() and mp.stat().st_size > 0:
                try:
                    import onnxruntime as ort  # type: ignore

                    opts = ort.SessionOptions()
                    opts.inter_op_num_threads = 1
                    opts.intra_op_num_threads = 1  # leave cores for STT on 4GB Pi
                    self._sess = ort.InferenceSession(str(mp), sess_options=opts)
                    self._names = [i.name for i in self._sess.get_inputs()]
                    self.reset()
                except Exception:
                    self._sess = None

    def reset(self) -> None:
        if self._sess is not None:
            import numpy as np  # type: ignore

            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, pcm16: bytes) -> bool:
        if not pcm16:
            return False
        if self._sess is not None:
            try:
                import numpy as np  # type: ignore

                samples = (
                    np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
                )
                CHUNK = 512  # silero v5 window
                probs = []
                for i in range(0, len(samples), CHUNK):
                    win = samples[i : i + CHUNK]
                    if len(win) < CHUNK:
                        win = np.pad(win, (0, CHUNK - len(win)))
                    feed = {
                        "input" if "input" in self._names else self._names[0]: win[
                            None, :
                        ]
                    }
                    if "state" in self._names and self._state is not None:
                        feed["state"] = self._state
                    if "sr" in self._names:
                        feed["sr"] = np.array(16000, dtype=np.int64)
                    outs = self._sess.run(None, feed)
                    probs.append(float(outs[0].flat[0]))
                    if len(outs) > 1 and "state" in self._names:
                        self._state = outs[1]
                return (max(probs) if probs else 0.0) >= self.threshold
            except Exception:
                pass
        n = len(pcm16) // 2
        s = struct.unpack(f"<{n}h", pcm16)
        rms = (sum(x * x for x in s) / max(n, 1)) ** 0.5 / 32768.0
        return rms > 0.015
