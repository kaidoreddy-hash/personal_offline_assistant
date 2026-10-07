"""STT: faster-whisper LOCAL-ONLY (never downloads at runtime), else mock.

Sovereignty rule: model_path must be a local dir/file already cached by
scripts/download_models.py. A HuggingFace model *name* is refused — that
would hit the network at runtime. Multilingual `tiny` (not the English-only
variant) so Hindi/Telugu/code-switch can re-enable without changing code.
"""

from __future__ import annotations
from pathlib import Path


def _local_model_dir(cfg: dict) -> Path | None:
    s = cfg.get("stt", {})
    mp = s.get("model_path", "models/stt-tiny")
    base = Path(__file__).resolve().parent.parent
    p = Path(mp) if Path(mp).is_absolute() else base / mp
    if p.is_dir() and any(p.iterdir()):
        return p
    if p.is_file() and p.stat().st_size > 0:
        return p
    return None


class STT:
    def __init__(self, cfg: dict):
        self.backend = "mock"
        self._model = None
        if cfg.get("stt", {}).get("backend", "auto") in ("auto", "faster-whisper"):
            local = _local_model_dir(cfg)
            if local is not None:  # local-only: no name -> no download
                try:
                    from faster_whisper import WhisperModel  # type: ignore

                    self._model = WhisperModel(
                        str(local),
                        device="cpu",
                        compute_type=cfg.get("stt", {}).get("compute_type", "int8"),
                    )
                    self.backend = "faster-whisper"
                except Exception:
                    self._model = None

    def transcribe(self, pcm16: bytes) -> str:
        if self._model is None:
            return ""  # mock: text arrives via WS events until STT cached
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
