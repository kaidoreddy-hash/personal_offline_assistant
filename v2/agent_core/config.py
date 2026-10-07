"""Config loader. stdlib-first; yaml optional."""

from __future__ import annotations
from pathlib import Path

_DEFAULTS = {
    "sample_rate": 16000,
    "wakeword": "hey tobi",
    "wakeword_onnx": "models/hey_jarvis_v0.1.onnx",
    "dormant_after_silence_s": 30,
    # ponytail: MUST mirror configs/pi5-4gb.yaml. Drift here silently disables
    # the VAD (0.6) and reverts to English-only STT. test_config_drift guards it.
    "vad": {
        "backend": "auto",
        "threshold": 0.2,
        "onnx_path": "models/silero_vad_v5.onnx",
    },
    "turn": {"backend": "auto", "min_speech_ms": 300},
    "stt": {
        "backend": "auto",
        "model": "tiny",
        "model_path": "models/stt-tiny",
        "whistle_path": "models/whistle.cact",
        "compute_type": "int8",
    },
    "llm": {
        "backend": "stub",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "local-slm",
    },
    "tts": {"backend": "auto", "voice": "models/en_US-lessac-low.onnx"},
    "logging": {"dir": "logs"},
}


def load(path: str | Path | None = None) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    p = (
        Path(path)
        if path
        else Path(__file__).resolve().parent.parent / "configs" / "pi5-4gb.yaml"
    )
    if p.exists():
        try:
            import yaml  # type: ignore

            user = yaml.safe_load(p.read_text()) or {}
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except ImportError:
            pass  # ponytail: no yaml on Pi minimal -> defaults hold
    return cfg
