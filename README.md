# ARIA - AI Companion Vision Agent

OpenCV 5 + AWS + LLM companion that sees visitors, converses with them, and
decides whether to enroll or alert. Three concurrent threads, no blocking.

## Quick Start
```bash
pip install -r requirements.txt
python setup_models.py   # fetches gitignored model files into models/
python run.py
# Keys: 1=Webcam 2=Phone 3=File 4=IPCAM | Q=Quit
```

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
