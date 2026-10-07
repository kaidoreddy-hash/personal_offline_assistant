# Tobi v2 — standalone full-duplex S2S (stub LLM, Pi 5 4GB target)

Battle-tested base, not reinvented. Vendors in `vendors/` are read-only clones (see `VENDOR_PINS.md`, never delete):
- `huggingface/speech-to-speech` — VAD→STT→LLM→TTS streaming pipeline, Realtime WS, offline recipe
- `asiff00/On-Device-S2S` — on-CPU Pi recipe
- `xcc-zach/xtalk` — full-duplex barge-in pattern

v2 swaps the LLM slot for a hardcoded streaming stub (`agent_core/llm_stub.py`).
Friend's offline SLM drops in via `llm.backend: openai-compatible` + `base_url` — no code change.

## Run (PC now, Pi later)

```
pip install -r requirements.txt
python app/serve.py            # http://127.0.0.1:8765
python app/serve.py --demo     # headless pipeline check
pytest tests/ -q               # 4 stub tests
python tools/mic_probe.py      # mic+VAD probe, writes logs/mic-*.jsonl
```

Open the page → Enable mic → say **“hey tobi”** → talk → stub streams back → talk over it to barge in → 30s silence → dormant.

## Layout

```
agent_core/  vad.py turn.py stt.py llm_stub.py tts.py duplex.py config.py logger.py
app/         serve.py (WS /ws) + web/index.html (no delay dropdown, no interrupt btn)
tools/       mic_probe.py log_dump.py
configs/     pi5-4gb.yaml
scripts/     pi_setup.sh dev_run.bat
logs/        mic-YYYYMMDD-HHMMSS-*.jsonl (every attempt, auto)
```

## Pi 5 wiring (ReSpeaker Lite 2-mic)

HAT on 40-pin → JST-PH2.0 speaker to `SPEAKER OUT` → 27W PSU → `scripts/pi_setup.sh` → `arecord -l` shows `seeed-2mic-voicecard` → pre-cache models online once → `HF_HUB_OFFLINE=1 python app/serve.py --host 0.0.0.0`. KY-040 → volume, MTS-202 → mic mute, LEDs mirror UI dot.
English-first; Telugu/Hindi re-enable via STT/TTS swap only.
