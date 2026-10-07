"""Regression: hardcoded defaults must mirror configs/pi5-4gb.yaml.

Drift here is silent and severe: a stale vad.threshold (0.6) makes the VAD deaf
and a stale stt.model (tiny.en) silently reverts to English-only.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from agent_core.config import _DEFAULTS, load


def test_defaults_match_yaml():
    yaml_cfg = load(BASE / "configs" / "pi5-4gb.yaml")
    drift = []
    for section in ("vad", "turn", "stt", "llm", "tts"):
        for k, v in _DEFAULTS[section].items():
            if k in yaml_cfg.get(section, {}) and yaml_cfg[section][k] != v:
                drift.append(
                    f"{section}.{k}: default={v!r} yaml={yaml_cfg[section][k]!r}"
                )
    assert not drift, "config drift:\n  " + "\n  ".join(drift)


def test_no_yaml_load_is_still_sane():
    """Simulates missing yaml file / missing pyyaml: must stay calibrated."""
    cfg = load("__missing__.yaml")
    assert cfg["vad"]["threshold"] == 0.2, "deaf VAD fallback"
    assert cfg["stt"]["model"] == "tiny", "English-only STT fallback"
    assert cfg["tts"]["voice"].endswith("lessac-low.onnx")


def test_wakeword_ignores_substring_words():
    from agent_core.duplex import FullDuplex

    dx = FullDuplex(load())
    for phrase in ("i met her in october", "autobiography is hard", "tobias left"):
        dx.state = "dormant"
        dx.on_partial(phrase)
        assert dx.state == "dormant", f"false wake on {phrase!r}"
    dx.state = "dormant"
    dx.on_partial("hey tobi what is the time")
    assert dx.state == "listening", "wake phrase must still work"
