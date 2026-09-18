# ARIA — Build Tasks (hand these to your coding agent one at a time)

Read `ARIA_ARCHITECTURE.md` first — these tasks implement that design. Each task below is written as a self-contained prompt.

**Development sequence: build and fully verify everything locally first.** Every phase through Phase 4 (perception, presence, conversation, audio) has zero AWS dependency — face detection/recognition, the state manager, and Sarvam STT/TTS all run entirely on your machine, no AWS account required. The only AWS-touching pieces are the LLM call in Task 8 and the logging in Task 11. Build both behind a swappable interface so the whole system is runnable and testable with no AWS account at all until the very last step, when you create one and flip a config value.

## Phase 0 — Fix current blockers

**Task 0: Get the model files onto this machine, and fix a model-version mismatch**
The app fails to detect faces because `models/` is missing entirely on whatever machine is running this — it's correctly gitignored (large binaries shouldn't go in git), but nothing downloads it, so a fresh clone or new environment has no model files at all. Confirmed directly: cloning the current repo shows no `models/` directory anywhere, and `VisionAgentApp.__init__()` in `run.py` constructs `Detector(...)` and `FaceEngine()` with no try/except around either, so this crashes hard at startup with `cv2.error: Can't read ONNX file` rather than failing silently.

Two-part fix:
1. **Fetch the models.** Three files are needed: `models/yolov5s.onnx` (the general-purpose detector in `detector.py`), plus `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` (the face pipeline in `face_engine.py`) from the OpenCV Zoo (`github.com/opencv/opencv_zoo`, under `models/face_detection_yunet/` and `models/face_recognition_sface/`). Since `models/` is correctly gitignored, add a small setup script (or a README section) that fetches these automatically — this isn't just about your own machine: the competition's final submission requires build instructions a judge can follow from a fresh clone, and right now a fresh clone can't run at all.
2. **Model filename — verified correct, do NOT rename.** `face_engine.py`'s `FACE_DET_MODEL` is `face_detection_yunet_2023mar.onnx` and must stay that way. An earlier draft of this task claimed a `face_detection_yunet_2026may.onnx` variant exists in the OpenCV Zoo and should replace it — that file does not exist; no such variant has ever been published there. `setup_models.py` fetches the real 2023mar file and the app loads it, so the mismatch this task warned about does not exist.

**Task 1: Fix the Sarvam STT/TTS connection error**
The error `create_connection() got an unexpected keyword argument 'extra_headers'` is a known breaking change: the `websockets` Python library renamed the `extra_headers` parameter to `additional_headers` starting in version 13. Check `pip show websockets` for the installed version, then find whichever code path (your own code or the Sarvam client library) passes `extra_headers` to a websockets connect call. Either rename it to `additional_headers`, or pin `websockets<13` in requirements if the Sarvam SDK itself hasn't updated yet. Confirm which fix applies before changing anything — don't guess blind.

**Task 2: Fix the local STT fallback (missing pyaudio)**
`pyaudio` isn't installed, so the fallback path in `speech_recognition`'s `Microphone()` fails after the Sarvam error. First try `pip install pyaudio` directly — modern releases ship precompiled wheels for most platforms. If that fails on this specific Python version (3.14, per the traceback, which is very new and may not have wheels published yet), replace the local-mic capture path with `sounddevice` instead — it has more reliable prebuilt-wheel support on Windows and is a drop-in alternative for reading raw audio from a microphone.

## Phase 1 — Perception upgrade

**Task 3: Add face detection via OpenCV 5's DNN face module**
Use `cv.FaceDetectorYN` (YuNet model) for detection: `cv.FaceDetectorYN.create(model_path, "", input_size, score_thresh, nms_thresh, top_k)`, then `.detect(frame)`, returning bounding boxes + 5 landmarks per face. Confirmed present and unchanged in OpenCV 5.0's own documentation.

**Task 4: Add face recognition/embedding matching**
Use `cv.FaceRecognizerSF` (SFace model) alongside the detector. Per detected face: `alignCrop()` → `.feature()` for an embedding → compare against every embedding in the registered-faces store via cosine similarity. OpenCV's own documented match threshold is 0.363 (cosine) / 1.128 (L2-norm) for "same person" — use these as starting points and tune against your own enrolled faces.

**Task 5: Add lightweight tracking + frame-skipping**
Don't re-run recognition on every frame for every face. Track faces frame-to-frame with simple centroid-distance matching; only trigger a fresh recognition call when a track is new or hasn't matched in a while. Downscale the frame before detection to keep this real-time on CPU.

## Phase 2 — Presence & state management

**Task 6: Build a presence state manager**
A component that consumes raw per-frame detections and emits clean, de-duplicated events: `person_entered`, `person_recognized(name)`, `person_unrecognized(track_id)`, `person_left`. Include a cooldown so an already-greeted person doesn't re-trigger the same event repeatedly within a configurable window (e.g. 5–10 minutes).

## Phase 3 — Conversation orchestrator

**Task 7: Event-triggered dialogue**
On `person_recognized` → generate a personalized greeting. On `person_unrecognized` → a friendly opener asking who they are (not an "intruder" framing — reads better in a demo and is the more honest posture). Once a conversation is active, ARIA responds to open-ended speech from anyone present, regardless of recognition status — recognition decides how a conversation starts, not whether ARIA responds at all.

**Task 8: Route conversation generation through AWS Bedrock**
Build this behind a simple interface — one function that takes the current event + conversation history and returns what ARIA says next — so the backend is swappable. Until an AWS account exists, point this at any other provider for local testing (a free-tier API key, or a local model). Once the account exists, swap the implementation to Bedrock, starting with a cheap tier (e.g. Amazon Nova Micro — cost per exchange is a small fraction of a cent, so a standard AWS sign-up credit covers extensive testing). Nothing else in the system needs to change when you make that swap. Document this clearly in the technical report as the primary "meaningful AWS component."

**Task 9: Design ARIA's personality/system prompt**
Write a system prompt establishing tone and behavioral rules — friendly and curious with unrecognized people, never hostile or accusatory. Keep the running conversation history short (last N turns) to control token costs.

## Phase 4 — Near-duplex audio

**Task 10: Add VAD-based interruption**
Integrate a lightweight voice-activity detector (e.g. Silero VAD — pip-installable, CPU-friendly) running continuously on the mic input. When it detects speech starting while ARIA's TTS is playing, immediately stop playback and switch to listening. Document this honestly as "interruptible turn-taking," not literal full duplex — true simultaneous duplex would additionally need acoustic echo cancellation and streaming (not batch) STT/TTS, a larger undertaking than this timeline supports.

## Phase 5 — AWS logging layer

**Task 11: Add DynamoDB + S3 logging**
Build this behind a logging interface too. Until AWS is set up, stub it locally — append to a local JSON-lines file, save snapshots to a local folder — so the rest of the system runs untouched. Once ready, swap in DynamoDB (track ID, name-or-unknown, timestamp, brief conversation summary) and S3 (flagged frames from unrecognized events) in parallel with the Bedrock swap in Task 8. This becomes both demo evidence and the "evaluation evidence, including failure cases" the final submission rules ask for.

## Phase 6 — Responsible-use safeguards

**Task 12: Consent-aware face registration**
Build the registration flow so it's deliberate and consent-based — registering a new "known" face should be an explicit action, never silent auto-enrollment from any detected face. State plainly in the technical report what happens to an unrecognized person's data: logged locally/in DynamoDB, not shared externally, and note retention if any.
