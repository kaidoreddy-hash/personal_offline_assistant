"""Tail the latest timestamped mic log."""

from pathlib import Path
import sys

d = Path(__file__).resolve().parent.parent / "logs"
files = sorted(d.glob("mic-*.jsonl"))
if not files:
    print("no logs yet")
    sys.exit(0)
p = files[-1] if len(sys.argv) < 2 else Path(sys.argv[1])
print(f"== {p} ==")
print(p.read_text()[-4000:])
