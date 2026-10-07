"""One runnable check for v2 stub pipeline. No frameworks beyond stdlib+pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_core.config import load
from agent_core.duplex import FullDuplex
from agent_core.llm_stub import HARDCODED_RESPONSE, stream_reply
from agent_core.turn import TurnDetector


def test_stub_streams_words():
    chunks = list(stream_reply("hello", {"llm": {"backend": "stub"}}))
    assert "".join(chunks) == HARDCODED_RESPONSE and len(chunks) > 5


def test_wake_final_reply_and_log(tmp_path=None):
    cfg = load()
    dx = FullDuplex(cfg)
    assert dx.state == "dormant"
    dx.on_partial("hey tobi")  # wake via transcript fallback
    assert dx.state == "listening"
    evs = dx.on_final("hello")
    text = "".join(e.data for e in evs if e.kind == "llm_chunk")
    assert text == HARDCODED_RESPONSE
    assert dx.log_path is not None and Path(dx.log_path).exists()
    assert "mic-" in Path(dx.log_path).name  # timestamped per requirement


def test_barge_in_cancels():
    cfg = load()
    dx = FullDuplex(cfg)
    dx.on_wakeword()
    dx.state = "responding"
    assert dx.on_barge_in()[0].data == "listening"


def test_turn_needs_speech_first():
    t = TurnDetector(load())
    assert t.update(False, "") == "continue"  # no premature end, no word heuristic
