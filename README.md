# ARIA - AI Companion Vision Agent

OpenCV 5 + AWS + LLM companion that sees visitors, converses with them, and
decides whether to enroll or alert. Three concurrent threads, no blocking.

## Quick Start
```bash
pip install -r requirements.txt
python setup_models.py   # fetches gitignored model files into models/
python run.py
# then open http://127.0.0.1:8080 (ARIA's UI runs in your browser)
```

> **Linux audio setup:** the voice stack needs PortAudio and espeak-ng as OS
> packages before `pip install -r requirements.txt` gives a working voice stack:
> `sudo apt install portaudio19-dev espeak-ng`. Windows/macOS wheels bundle
> PortAudio; espeak-ng is only needed for the pyttsx3 offline TTS fallback
> (Windows uses SAPI5 instead).

## Face Authorization Security

Face recognition thresholds are **calibrated, not guessed**. Baseline data lives
in `baseline-recognition.json`; pair distributions and the FAR/FRR curve are
written by:

```bash
python build_baseline.py        # per-person intra / cross-person similarity
python calibrate_threshold.py   # genuine vs impostor sweep -> FAR/FRR/EER
```

Tunables (env vars):

```bash
ARIA_RECOGNITION_THRESHOLD=0.36  # min cosine similarity to authorize (calibrate before changing)
ARIA_QUALITY_MIN=0.30            # poor quality REJECTS - it never lowers the bar
ARIA_MARGIN_MIN=0.05             # best must beat 2nd-best identity by this much
ARIA_ENROLL_MIN_SAMPLES=3        # independent captures required for enrollment
ARIA_FACE_TELEMETRY=1            # [FACE] decision lines per recognition round
```

Security invariants (enforced by `test_recognition_security.py`):

- A track ID is **not** proof of identity - revoked tracks drop to "unknown".
- Authorization requires fresh evidence; misses revoke in 2 rounds.
- Detection-to-track assignment is one-to-one (Hungarian, not greedy).
- Runtime recognition **never** writes `faces/*.npy`; only `enroll_identity()`
  does, and only after duplicate/diversity/quality validation. The old
  "one frame x 5" enrollment is impossible by construction.

## Face Pattern HUD
Unrecognized visitors get a MediaPipe Face Landmarker mesh drawn over their
actual face boundary (contours style); the moment a face is recognized the
pattern disappears automatically. Tunables (env vars):

```bash
ARIA_PATTERN_STYLE=contours   # contours | tesselation (denser wireframe look)
ARIA_PATTERN_EVERY=2          # re-infer every Nth frame (higher = cheaper)
ARIA_PATTERN_MAX_FACES=4      # simultaneous meshes
```
Requires `mediapipe` (auto-installed via requirements.txt) and
`models/face_landmarker.task` (fetched by `setup_models.py`).

## Architecture (Full-Duplex)
```
┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐
│ VisionThread │──▶│ vision_events Queue │──▶│ ConvThread  │
│ (OpenCV 5)   │    │                     │    │ (LLM+Voice) │
│ • YOLOv5s    │    │ queue.Queue (safe)  │    │ • Bedrock   │
│ • FaceEngine │    │ threading.Lock      │    │ • TTS/STT   │
└─────────────┘    └─────────────────────┘    └─────────────┘
       │                                          │
       │ frame buffer (shared)            actions (enroll/alert)
       ▼                                          ▼
┌─────────────┐                         ┌─────────────┐
│ MainThread  │◀────────────────────────│             │
│ (render)    │   state events           │ Alerter     │
│ cv2.imshow  │                          │ (SNS)       │
└─────────────┘                          └─────────────┘
```

## Intelligence Tiers (tried in order)
1. **AWS Bedrock Claude** — set `AWS_BEDROCK_REGION` → real LLM conversation
2. **Local Ollama** — `pip install ollama` + run → offline LLM
3. **Rule-based fallback** — keyword matching, no dependencies
```
# pro-to-type
