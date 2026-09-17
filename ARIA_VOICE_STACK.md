# ARIA Voice Stack — Talkative Companion Upgrade (2026-09)

How ARIA's conversation layer works after the talkative upgrade, modeled on
OpenAI's GPT-Live architecture (their docs: voice-agents + realtime + VAD
guides) adapted to a local, laptop-speaker, Sarvam-backed system.

## Before → After

| Capability | Before | After |
|---|---|---|
| Dialogue replies | **Never spoken** (text discarded in `_dialogue_loop`) | Spoken, sentence-by-sentence |
| TTS latency | Whole utterance buffered before playback | Plays as chunks arrive (streaming `OutputStream`) |
| First audible word | After full LLM answer + full TTS download | ~1 sentence after streaming starts |
| Turn detection | Fixed 8–12s record windows | VAD: starts at speech onset, ends at ~600ms quiet |
| Echo safety | N/A (half-duplex) | Mic suppressed while ARIA speaks (pseudo-duplex) |
| Deep questions | Same slow path as small talk | Voice brain answers small talk; deep questions **delegated** (`[DELEGATE]`) to the backend brain with a filler while it works |
| Long silences | Session reset after ~1 miss | Gentle nudges (max 2), session stays open |
| Initiative | None — only reactive | Proactive grounded remarks (rate-limited) |

## Data flow (one conversational turn)

```
MicVAD thread          conversation thread            SpeechQueue thread
─────────────          ───────────────────            ──────────────────
energy VAD ──utter──▶ listen(mic)                     ◀── enqueue_speech("Hi again!")
                       │                                   │
                       ▼                                   ▼
                  VoiceSession.run_turn()           speak() → streaming TTS
                       │                                   ▲
        ┌──────────────┴──────────────┐                    │
        ▼                             ▼                    │
  VoiceBrain (fast)          [DELEGATE]? ──▶ backend brain  │
  1-2 spoken sentences       no → done       (stream_response) with
  streamed sentence-         yes → filler    "Okay, give me a second…"
  by-sentence to queue                       then streams the full answer
```

## Components

| Module | Role |
|---|---|
| `mic_vad.py` | Always-on mic thread: energy VAD, adaptive noise floor, utterance = onset + ≥600ms quiet, echo guard while `speaking` is set |
| `voice.py` | `_listen_vad` (transcribes VAD-captured PCM), streaming `_speak_sarvam`, `SpeechQueue` (sentence worker + `wait_speech_done`), `set_speaking_callback` (echo guard hook), generation-counter barge-in flushes queued sentences |
| `agent.py` | `VoiceBrain` (speaking-behavior prompt, `[DELEGATE]` protocol), `VoiceSession` (turn engine: sentence streaming, fillers, backend delivery + cache, nudges), `VisionContext` (per-frame "what ARIA sees" summary) |
| `companion.py` | `ProactiveEngine`: memory-grounded follow-ups + LLM observations, min-interval 90s, ≤6/hour, silent in empty rooms |
| `run.py` | Wires it together; `_dialogue_loop` now speaks every reply; vision thread feeds `VisionContext` each frame |

## GPT-Live concepts → ARIA implementation

- **Live model / speaking behavior** → `VoiceBrain` + `VOICE_SYSTEM_PROMPT` (1–2 sentences, no markdown, at most one question)
- **Backend delegation (client mode)** → `VoiceSession._backend_deliver` calling `VisionAgent.stream_response` (memory + tools + personality)
- **server_vad / semantic_vad** → energy VAD in `MicVAD` (silence-chunking); semantic end-of-turn is future work
- **Fillers while reasoning** → `VoiceBrain.filler()` spoken if no sentence within 1.2s, max 2 per turn
- **Acknowledgment ≠ answer** → fillers/fillers use a separate enqueue path; useful answer latency measured separately in logs

## Tuning knobs (env)

- `ARIA_VAD_SILENCE_MS` (not yet wired — constructor args on `MicVAD`) — end-of-turn quiet, default 600ms
- `ARIA_PATTERN_EVERY`, `ARIA_PATTERN_STYLE` — unchanged pattern renderer knobs

## Known limits / next steps

- No true AEC: barge-in works via generation counters, but ARIA can't hear
  you while she's speaking (mic is suppressed). Fine for laptop speakers.
- Saaras STT transcribes a captured utterance in one request; streaming
  partials (`stream_transcript`) remain unused — captions are future work.
- `semantic_vad`-style eagerness (let users trail off "ummm…") would need a
  tiny classifier or LLM ping — candidate next upgrade.
