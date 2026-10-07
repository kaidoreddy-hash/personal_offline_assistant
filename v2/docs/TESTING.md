# Testing guide — tobi v2

## Automated (no mic needed)

```
python tools/check_offline.py     # sovereignty gate: all models load with sockets BLOCKED
python -m pytest tests/ -q        # 26 tests: config drift, VAD, turn, E2E, WS, store, tools
python tools/e2e_offline.py       # fixture wav -> transcript + reply wav (data/e2e_reply.wav)
```

Fixtures are Piper-synthesized (tools/gen_fixtures.py regenerates them).

Known fixture limit (pinned by a test, do not "fix" blindly): Piper gives every clip final
declarative prosody, so a truncated sentence still scores "complete" on Smart-Turn. Real humans
trail off — verify that case by ear below. The important automated signals: "Umm" HOLDS the turn,
completes complete, noise/silence produce nothing.

## Manual (your real mic)

1. `python -m server.ws_server --config configs/desktop.yaml` → open http://127.0.0.1:8000
2. Click "enable microphone" (one click; browser AEC makes barge-in safe).
3. Acceptance list:
   - [ ] page shows "dormant" + wake hint
   - [ ] say "Hey Jarvis" → greeting plays, pill turns listening
   - [ ] speak → live partial transcript appears while talking
   - [ ] finish a sentence → reply streams as text AND audio (~1s or less)
   - [ ] mid-sentence pause → it WAITS (smart-turn), then continues when you continue
   - [ ] talk over the reply → audio stops within ~300ms, back to listening
   - [ ] stay silent 90s → dormant; wake word re-activates
   - [ ] music/typing noise → no false wake, no phantom utterances
   - [ ] "what's the weather in Hyderabad" → online weather, spoken + shown
   - [ ] "send me the summary on telegram" → message on your phone (set TOBI_TG_TOKEN/TOBI_TG_CHAT first)
   - [ ] meeting mode → purple state, segments listed → end meeting → speaker labels + summary
   - [ ] logs/session_*/events.jsonl exists for EVERY attempt (wake scores, turns, latencies)

## Debugging a bad mic session

Every session writes logs/session_<ts>/events.jsonl (+ utterance wavs). Grep the jsonl for
`wake`, `turn_check`, `final`, `barge_in`, `error`. If VAD never fires, run tools/mic_probe
style checks: check your OS default input device first; the browser uses its own input picker.

## Benchmarks

Desktop (this laptop, measured): EOT → first audio = 562ms with the stub brain; Smart-Turn
~50-60ms per check; Silero ~1ms/frame. Pi targets: EOT → voice < 2.5s; record numbers in
docs/pi5_deploy.md when the unit arrives.
