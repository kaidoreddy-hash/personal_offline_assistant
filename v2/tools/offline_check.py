"""Offline sovereignty check. Fails on ANY cloud/runtime-download dependency."""

from __future__ import annotations
import re, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUNTIME = list((BASE / "agent_core").glob("*.py")) + [
    BASE / "app" / "serve.py",
    BASE / "app" / "web" / "index.html",
]

BANNED = [
    (r"SpeechRecognition|webkitSpeechRecognition", "browser cloud STT (Google)"),
    (r"torch\.hub\.load", "runtime model download via torch.hub"),
    (r"api\.openai\.com", "OpenAI cloud"),
    (r"googleapis\.com|google\.com/speech", "Google cloud speech"),
    (r"amazonaws\.com.*transcribe|azure.*speech", "cloud speech vendor"),
    (
        r"WhisperModel\(\s*[\"'](tiny|base|small|large)",
        "faster-whisper model NAME (downloads); must be local path",
    ),
]


def main() -> int:
    bad = 0
    for f in RUNTIME:
        text = f.read_text(errors="replace")
        for pat, why in BANNED:
            if re.search(pat, text):
                print(f"FAIL {f.relative_to(BASE)}: {why} [{pat}]")
                bad += 1
    # config sovereignty: STT must be multilingual tiny (not tiny.en), LLM loopback-only
    cfg = (BASE / "configs" / "pi5-4gb.yaml").read_text()
    if "tiny.en" in cfg:
        print(
            "FAIL pi5-4gb.yaml: tiny.en is English-only; use multilingual tiny for Hindi/Telugu path"
        )
        bad += 1
    m = re.search(r"base_url:\s*(\S+)", cfg)
    if m and not re.search(
        r"127\.0\.0\.1|localhost|192\.168\.|10\.|172\.16\.", m.group(1)
    ):
        print(f"FAIL pi5-4gb.yaml: llm.base_url must be loopback/LAN, got {m.group(1)}")
        bad += 1
    # offline flags must be set by serve.py by default
    serve = (BASE / "app" / "serve.py").read_text()
    if "HF_HUB_OFFLINE" not in serve:
        print("FAIL serve.py: must set HF_HUB_OFFLINE=1 by default")
        bad += 1
    print(
        "offline-check: FAIL"
        if bad
        else "offline-check: PASS (no cloud deps, local-only models)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
