"""VAD: Silero v5 if torch present, else energy fallback. No UI knobs."""

from __future__ import annotations
import os
import struct


class VAD:
    def __init__(self, cfg: dict):
        v = cfg.get("vad", {})
        self.threshold = float(v.get("threshold", 0.6))
        self._silero = None
        # ponytail: offline-first — torch.hub.load() downloads on cache miss,
        # so only try it on an explicit pre-cache run (V2_ALLOW_NET=1).
        if (
            v.get("backend", "auto") in ("auto", "silero-v5")
            and os.environ.get("V2_ALLOW_NET") == "1"
        ):
            try:
                import torch  # type: ignore

                self._silero, _ = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", trust_repo=True
                )
            except Exception:
                self._silero = None

    def is_speech(self, pcm16: bytes) -> bool:
        if self._silero is not None:
            try:
                import torch, numpy as np  # type: ignore

                a = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
                with __import__("torch").no_grad():
                    p = float(self._silero(torch.from_numpy(a), 16000).item())
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
