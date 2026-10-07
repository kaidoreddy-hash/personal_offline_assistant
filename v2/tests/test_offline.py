"""Sovereignty regression: no cloud STT/LLM/VAD at runtime. Offline-only."""

import subprocess, sys
from pathlib import Path


def test_offline_check_passes():
    base = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, str(base / "tools" / "offline_check.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_llm_refuses_cloud_url():
    import sys as _s

    _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent_core.llm_stub import stream_reply

    out = "".join(
        stream_reply(
            "hi",
            {
                "llm": {
                    "backend": "openai-compatible",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        )
    )
    assert "refused non-local" in out


def test_stt_never_uses_model_name():
    src = (Path(__file__).resolve().parent.parent / "agent_core" / "stt.py").read_text()
    assert "tiny.en" not in src and 'WhisperModel("tiny' not in src
