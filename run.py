"""
VisionAgentApp — Full-duplex AI companion architecture.

Threads:
  1. VisionThread — runs OpenCV pipeline (YOLO + face detection) continuously
  2. VoiceThread — manages conversation: TTS speak + STT listen + LLM think
  3. Main thread — renders video overlay + handles input switching

Communication via signals/events (thread-safe queues).
"""

import os
import sys
import time
import threading
import queue
import json
import datetime

# Auto-load .env for direct `python run.py` execution
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    for line in open(_env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import cv2
import numpy as np

from detector import Detector
from reasoner import Reasoner
from alerter import Alerter
from face_engine import FaceEngine
from tracker import FaceTracker
from presence import PresenceManager
from input_source import InputManager, WebcamSource, IPCameraSource, FileSource
from conversation import ConversationManager
from agent import VisionAgent
from voice import SarvamVoice
from context_memory import PersonMemory, SessionManager


CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "models/yolov5s.onnx"),
    "phone_url": os.getenv("PHONE_CAM_URL", ""),
    "ipcam_url": os.getenv("IP_CAM_URL", ""),
    "file_source": os.getenv("VIDEO_SOURCE", "samples/face_test.jpg"),
}

# Words that mean "no" — anything else is treated as consent to enroll
_REFUSAL_WORDS = {"no", "nope", "nah", "never", "later", "not", "don't", "dont", "leave", "stop", "privacy"}

# Placeholder names that must never be enrolled as a person's identity
_INVALID_NAMES = {
    "", "unknown", "uh", "um", "my", "i", "the", "name", "is", "it's", "its",
    "im", "i'm", "this", "yes", "yeah", "yep", "ok", "okay", "sure", "thanks",
    "thank", "hello", "hi", "hey", "no", "not", "don't", "dont", "sorry",
    "very", "just", "so", "really", "here", "called", "and", "but",
}


class VisionAgentApp:
    def __init__(self):
        self.detector = Detector(CONFIG["model_path"])
        self.face_engine = FaceEngine()
        self.reasoner = Reasoner(face_engine=self.face_engine)
        self.reasoner.set_authorized(set(self.face_engine.known_faces.keys()))
        self.alerter = Alerter(enabled=bool(os.getenv("ALERT_SNS_TOPIC_ARN")))

        self.agent = VisionAgent()
        self.voice = SarvamVoice()
        self.memory: dict[str, PersonMemory] = {}

        # Thread-safe event queues
        self.vision_events: queue.Queue = queue.Queue()
        self.action_events: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # Restore session state
        saved = SessionManager.load_current_state()
        if saved:
            self.memory.update(saved)
            print(f"[Session] Resumed {len(saved)} person memory(s)")

        # Input management
        self.manager = InputManager()
        self.manager.register("webcam", WebcamSource(0))
        self.manager.register("phone", IPCameraSource(CONFIG["phone_url"], "Phone Camera"))
        self.manager.register("file", FileSource(CONFIG["file_source"]))
        if CONFIG["ipcam_url"]:
            self.manager.register("ipcam", IPCameraSource(CONFIG["ipcam_url"], "IP Camera"))

        # Latest frame + analysis (shared between vision thread and render thread)
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_decision = None
        self._status_text = ""

        # Face tracking + presence management (architecture: separate seeing from thinking)
        self.tracker = FaceTracker(self.face_engine)
        self.presence = PresenceManager(self.vision_events)

    def run(self):
        cv2.namedWindow("ARIA — Full-Duplex Vision Agent")
        cv2.setMouseCallback("ARIA — Full-Duplex Vision Agent",
                             self.manager.handle_mouse, {"frame_w": 640})
        self.manager.switch_to("webcam")

        vision_t = threading.Thread(target=self._vision_loop, daemon=True)
        voice_t = threading.Thread(target=self._conversation_loop, daemon=True)
        vision_t.start()
        voice_t.start()

        print("ARIA ready. Keys: 1=Webcam 2=Phone 3=File 4=IPCAM. Q=Quit")

        while not self._stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
                decision = self._latest_decision

            if frame is None:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    self._stop.set()
                continue

            self._render(frame, decision)

            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
            if key == ord("q"):
                self._stop.set()

        vision_t.join(timeout=2)
        voice_t.join(timeout=2)
        cv2.destroyAllWindows()

    def _vision_loop(self):
        """Continuously captures + analyzes frames using tracker + presence manager.

        Architecture (per ARIA_ARCHITECTURE.md): OpenCV does ALL seeing.
        - YOLO detects people (visualization + awareness)
        - FaceEngine detects faces (downscaled for CPU speed)
        - FaceTracker matches faces frame-to-frame, skips redundant recognition
        - PresenceManager emits clean, de-duplicated state-transition events
        """
        fps_buf = []
        while not self._stop.is_set():
            ret, frame = self.manager.read()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            t0 = time.perf_counter()

            # YOLO object detection (people, objects) — for visualization
            detections = self.detector.detect(frame)

            # Face tracking (with frame-skipping + downscale)
            face_results = self.tracker.update(frame, downscale_factor=0.5)

            # Run recognition only on faces that need it (new or stale tracks)
            if any(fr.get("needs_recognition") for fr in face_results):
                self.tracker.recognize_faces(frame)
                face_results = self.tracker.update_results()

            # Presence: emit de-duplicated events (person_entered, person_recognized, etc.)
            self.presence.update(face_results)

            # Clean up lost faces
            self.tracker.cleanup_left_faces()

            # Render
            for d in detections:
                self.detector.draw(frame, d)
            for fr in face_results:
                self._draw_face(frame, fr)

            fps_buf.append(time.perf_counter() - t0)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            avg_dt = sum(fps_buf) / len(fps_buf) if fps_buf else 0.1
            self._status_text = f"{1/avg_dt:.1f} FPS | faces={len(face_results)} | tracks={self.tracker.track_count}"

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_decision = {
                    "face_results": face_results,
                    "should_act": any(not fr["authorized"] for fr in face_results),
                    "total": len(detections),
                }

            # Periodic frame summary (every 3s)
            now = time.time()
            if hasattr(self, "_last_summary_time"):
                if now - self._last_summary_time >= 3.0:
                    self._last_summary_time = now
                    self.vision_events.put({
                        "type": "frame_summary",
                        "detections": len(detections),
                        "faces": len(face_results),
                        "authorized": sum(1 for f in face_results if f["authorized"]),
                    })
            else:
                self._last_summary_time = now

    # ------------------------------------------------------------------
    # Shared conversation helpers
    # ------------------------------------------------------------------

    def _speak(self, text: str):
        """Speak and actually wait for playback to finish.

        Every caller previously used speak_async + a guessed sleep, which is
        how the greeting got cut off and how TTS overlapped with listen().
        """
        if text and text.strip():
            self.voice.speak(text)

    def _latest_frame_now(self):
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _latest_stable_face(self, min_quality: float = 0.4):
        """Return (frame, face) for the current best visible face, or (None, None).

        Requires the same face to be detected on two consecutive quick reads —
        guards against enrolling someone from a transient/partial frame.
        """
        for _ in range(2):
            frame = self._latest_frame_now()
            if frame is None:
                time.sleep(0.2)
                continue
            faces = self.face_engine.detect(frame)
            if len(faces) == 0:
                time.sleep(0.4)
                continue
            face = max(faces, key=lambda f: f[2] * f[3])
            quality = self.face_engine._face_quality(face, frame)
            if quality >= min_quality:
                return frame, face
            time.sleep(0.4)
        return None, None

    @staticmethod
    def _extract_name(utterance: str) -> str | None:
        """Pull a plausible person name out of a raw utterance, or None.

        B3 fix: never enroll "Hey" (first token) or "my name is" (pre-name
        filler) as someone's identity.
        """
        text = (utterance or "").strip().strip(".,!?").strip()
        if not text:
            return None
        low = text.lower()

        for marker in ("my name is", "i am called", "call me", "i'm", "im", "i am", "this is"):
            if marker in low:
                tail = text[low.index(marker) + len(marker):].strip(" .,!?")
                words = [w for w in tail.split() if w.strip(" .,!?")]
                # Skip filler words and capitalized markers embedded in phrases
                while words and words[0].lower().strip(" .,!?") in _INVALID_NAMES | {"a", "an", "just", "called"}:
                    words.pop(0)
                if words:
                    return words[0].strip(" .,!?")
                return None

        # No marker: accept a lone word as the name only if it isn't filler
        words = text.split()
        if len(words) == 1 and words[0].lower() not in _INVALID_NAMES:
            return words[0]
        return None

    def _enroll_with_name(self, conv: ConversationManager, name: str, face_data: dict):
        """Enroll the visitor under `name`, capturing templates from FRESH frames.

        B4 fix: the old code embedded the SAME frame three times, producing
        three near-identical templates — useless for multi-angle robustness.
        Each capture here re-reads the live camera and skips near-duplicates.
        """
        for attempt in range(3):
            frame, face = self._latest_stable_face()
            if frame is not None and face is not None:
                emb = self.face_engine.embed(frame, face)
                if emb is not None and len(emb) > 0:
                    # Skip templates that are near-identical to ones we already have
                    existing = self.face_engine.known_faces.get(name, [])
                    if not any(float(np.dot(emb, e)) > 0.995 for e in existing):
                        self.face_engine.add_template(name, emb)
                        time.sleep(0.8)  # let the person move naturally between captures
                    break
            time.sleep(0.5)

        templates = len(self.face_engine.known_faces.get(name, []))
        if templates == 0:
            self._speak("I couldn't get a clear look at your face. Let's try that again in a moment.")
            conv.reset()
            return

        self.reasoner.set_authorized(set(self.face_engine.known_faces.keys()))

        mem = self.agent._memory_for(name)
        mem.persona.name = name
        mem.persona.relationship_tier = "trusted"
        mem.persona.purpose = face_data.get("purpose") or "enrolled visitor"
        mem.persona.tags.append("enrolled")
        mem.add_interaction("system", f"Enrolled as {name} with {templates} templates")
        mem.save()

        self.memory[name] = mem
        # If we had been talking to this person under a generic identity, merge forward
        if conv.current_identity and conv.current_identity in self.agent.memory and conv.current_identity != name:
            del self.agent.memory[conv.current_identity]
        conv.current_identity = name

        conv.reset()
        self._speak(f"Pleased to meet you, {name}! You're now authorized — I'll recognize you next time.")

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    def _conversation_loop(self):
        """Consumes presence events, drives conversation via voice + LLM.

        Events from PresenceManager: person_entered, person_recognized,
        person_unrecognized, person_left.
        Uses ConversationManager for state machine + agent for LLM reasoning.
        """
        conv = ConversationManager(
            self.agent, self.voice, self.face_engine,
            vision_queue=self.action_events,
            frame_provider=self._latest_frame_now,
        )

        while not self._stop.is_set():
            try:
                event = self.vision_events.get(timeout=0.5)
            except queue.Empty:
                continue

            evt_type = event.get("type", "")

            if evt_type == "person_recognized":
                # B5 fix: greet known people, then LISTEN — they get a dialogue,
                # not a monologue followed by silence.
                name = event.get("name", "friend")
                track_id = event.get("track_id", "")
                face_data = {
                    "authorized": True,
                    "face_bbox": event.get("face_bbox", (0, 0, 0, 0)),
                    "score": event.get("score", 0.0),
                }
                self.presence.mark_greeted(track_id)

                response, action = self.agent.think(name, face_data)
                self._speak(response)
                self._dialogue_loop(conv, name, face_data)

            elif evt_type == "person_unrecognized":
                # B1 fix: the greeting is spoken exactly once. start_for only
                # builds and returns the text now; playback happens here.
                face_data = {
                    "authorized": False,
                    "face_bbox": event.get("face_bbox", (0, 0, 0, 0)),
                    "score": event.get("score", 0.0),
                }
                self.presence.mark_greeted(event.get("track_id", ""))

                greeting = conv.start_for("unknown", face_data)
                self._speak(greeting)
                self._dialogue_loop(conv, "unknown", face_data)

            elif evt_type == "person_left":
                conv.reset()
                self.voice.speak_async("Goodbye! Come back soon.")

            elif evt_type == "frame_summary":
                self._status_text = (
                    f"Vision: {event['detections']} dets, "
                    f"{event['faces']} faces, "
                    f"{event['authorized']} authorized"
                )

    def _dialogue_loop(self, conv: ConversationManager, identity: str, face_data: dict):
        """Shared listen/respond loop for recognized AND unknown visitors.

        B5 fix: recognized visitors previously never reached a listen() call,
        so anything they said after the greeting fell into the void.
        """
        if conv.current_identity is None:
            conv.current_identity = identity

        while conv.state != "IDLE" and not self._stop.is_set():
            transcript = self.voice.listen(timeout=12, phrase_limit=8)
            if not transcript:
                # No reply — end politely instead of hanging in the state machine
                if conv.state != "ENROLLING":
                    conv.reset()
                    break
                continue

            response, action = conv.handle_response(transcript, conv.current_identity, face_data)

            if action == "enroll":
                conv.state = "ENROLLING"
                # B3 fix: derive the name from what they actually said; if we
                # can't, ask instead of enrolling "Hey" as a person.
                name = self._extract_name(transcript)
                if name is None:
                    self._speak("And what should I call you?")
                    name_reply = self.voice.listen(timeout=8, phrase_limit=5)
                    name = self._extract_name(name_reply or "")
                if name is None:
                    self._speak("No worries — we can skip that for now.")
                    conv.reset()
                else:
                    face_data["purpose"] = transcript
                    self._enroll_with_name(conv, name, face_data)
                break
            elif action == "alert":
                self.alerter.fire({
                    "reason": "unknown_visitor_refused_enrollment",
                    "transcript": transcript,
                })
                self.voice.speak_async("I'm escalating this to security.")
                conv.reset()
                break
            # action == "ask": loop continues — listen for their next line

    # ------------------------------------------------------------------

    def _draw_face(self, frame, fr):
        color = (0, 255, 0) if fr["authorized"] else (0, 0, 255)
        x, y, w, h = fr["face_bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if fr.get("landmarks"):
            pts = np.array(fr["landmarks"]).reshape(-1, 2)
            for px, py in pts:
                cv2.circle(frame, (int(px), int(py)), 3, color, -1)
        label = f"{fr['identity']} ({fr['distance']:.2f})"
        cv2.putText(frame, label, (x, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _render(self, frame, decision):
        frame_disp = frame.copy()
        if self._status_text:
            cv2.putText(frame_disp, self._status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        frame_disp = self.manager.draw_controls(frame_disp)
        cv2.imshow("ARIA — Full-Duplex Vision Agent", frame_disp)

    def _handle_key(self, key):
        key_map = {"1": "webcam", "2": "phone", "3": "file", "4": "ipcam"}
        if chr(key) in key_map and chr(key) in self.manager._sources:
            self.manager.switch_to(chr(key))


if __name__ == "__main__":
    app = VisionAgentApp()
    app.run()
