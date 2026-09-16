# ARIA - AI Companion Vision Agent

OpenCV 5 + AWS + LLM companion that sees visitors, converses with them, and
decides whether to enroll or alert. Three concurrent threads, no blocking.

## Quick Start
```bash
pip install -r requirements.txt
python run.py
# Keys: 1=Webcam 2=Phone 3=File 4=IPCAM | Q=Quit
```

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
