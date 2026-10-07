"""Every mic attempt auto-produces a timestamped debug log file."""

from __future__ import annotations
import json, time
from pathlib import Path


def _logdir(cfg: dict) -> Path:
    d = Path(cfg.get("logging", {}).get("dir", "logs"))
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_attempt(cfg: dict, source: str = "mic") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    ms = int((time.time() % 1) * 1000)
    p = _logdir(cfg) / f"mic-{ts}-{ms:03d}-{source}.jsonl"
    p.write_text(
        json.dumps({"t": time.time(), "event": "attempt_start", "source": source})
        + "\n"
    )
    return p


def log(path: Path, event: str, **fields) -> None:
    rec = {"t": time.time(), "event": event, **fields}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
