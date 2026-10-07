"""STT: Whistle fast-path (EN) -> faster-whisper int8 multilingual. LOCAL-ONLY.

Greedy decode (beam_size=1) for latency; condition_on_previous_text=False so
code-switching mid-utterance doesn't contaminate the next window.
"""

from __future__ import annotations
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _local(name: str) -> Path | None:
    p = Path(name) if Path(name).is_absolute() else BASE / name
    return p if p.exists() and (p.is_dir() or p.stat().st_size > 0) else None


class STT:
    def __init__(self, cfg: dict):
        s = cfg.get("stt", {})
        want = s.get("backend", "auto")
        self.backend = "none"
        self._whistle = None
        self._fw = None
        if want in ("auto", "whistle"):
            wp = _local(s.get("whistle_path", "models/whistle.cact"))
            if wp:
                try:
                    from needle import Whistle  # type: ignore

                    self._whistle = Whistle(weights=str(wp))
                    self.backend = "whistle"
                except Exception:
                    self._whistle = None
        if want in ("auto", "faster-whisper"):
            p = _local(s.get("model_path", "models/stt-tiny"))
            if p:
                try:
                    from faster_whisper import WhisperModel  # type: ignore

                    self._fw = WhisperModel(
                        str(p),
                        device="cpu",
                        compute_type=s.get("compute_type", "int8"),
                        cpu_threads=4,
                    )
                    self.backend = "whistle+fw" if self._whistle else "faster-whisper"
                except Exception:
                    self._fw = None

    def transcribe(self, pcm16: bytes, language: str | None = None) -> str:
        if not pcm16:
            return ""
        if self._whistle is not None and (
            language is None or language in ("en", "de", "fr", "es", "it", "nl", "pl")
        ):
            try:
                import numpy as np, tempfile, wave  # type: ignore

                a = np.frombuffer(pcm16, dtype=np.int16)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    with wave.open(f.name, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes(a.tobytes())
                    r = (
                        self._whistle.transcribe(f.name)
                        if hasattr(self._whistle, "transcribe")
                        else None
                    )
                    if isinstance(r, dict) and r.get("text"):
                        return str(r["text"]).strip()
            except Exception:
                pass
        if self._fw is not None:
            try:
                import numpy as np  # type: ignore

                a = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
                segs, _ = self._fw.transcribe(
                    a,
                    language=language,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                return " ".join(s.text for s in segs).strip()
            except Exception:
                return ""
        return ""
