"""Full-duplex pipeline tests: state machine, wake, barge-in, turn, log."""

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


def test_wake_final_reply_and_log():
    dx = FullDuplex(load())
    assert dx.state == "dormant"
    evs_wake = dx.on_partial("hey tobi")  # transcript wake fallback
    assert dx.state == "listening"
    assert any(e.kind == "state" and e.data == "listening" for e in evs_wake)

    evs_final = dx.on_final("What time is it?")
    assert (
        "".join(e.data for e in evs_final if e.kind == "llm_chunk")
        == HARDCODED_RESPONSE
    )
    assert dx.state == "responding", "must stay responding until playback_done"

    assert dx.on_playback_done()[0].data == "listening"
    assert dx.log_path is not None and Path(dx.log_path).exists()
    assert "mic-" in Path(dx.log_path).name  # timestamped per requirement


def test_barge_in_cancels_reply():
    dx = FullDuplex(load())
    dx.on_wakeword()
    dx.state = "responding"
    evs = dx.on_barge_in()
    assert dx.state == "listening"
    assert any(e.kind == "barge_in" for e in evs)


def test_turn_needs_speech_first():
    t = TurnDetector(load())
    assert t.update(False, "") == "continue"  # no premature end
    for _ in range(4):
        assert t.update(True, "hello") == "continue"
    assert t.update(False, "hello") == "continue"  # tail still filling


def test_vad_is_stateful_silero():
    from agent_core.vad import VAD

    v = VAD(load())
    assert v._sess is None or v._state is not None, "silero needs recurrent state"
