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

        # Face tracking — prevents re-greeting identified faces
        self._face_trackers: dict[str, dict] = {}
        self._greeted_identities: set = set()

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
        """Continuously captures + analyzes frames, sends events to conversation."""
        fps_buf = []
        while not self._stop.is_set():
            ret, frame = self.manager.read()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            t0 = time.perf_counter()
            detections = self.detector.detect(frame)
            decision = self.reasoner.evaluate(detections, frame)
            face_results = decision.get("face_results", [])

            for d in detections:
                self.detector.draw(frame, d)
            for fr in face_results:
                self._draw_face(frame, fr)

            fps_buf.append(time.perf_counter() - t0)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            avg_dt = sum(fps_buf) / len(fps_buf) if fps_buf else 0.1
            self._status_text = f"{1/avg_dt:.1f} FPS | faces={len(face_results)} | act={decision['should_act']}"

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_decision = decision

            # --- Vision → Conversation signal ---
            # Send event for unknown faces not yet greeted, and skip low-quality/blurry faces
            for fr in face_results:
                if not fr["authorized"]:
                    person_key = fr["identity"]
                    already_greeted = self._check_greeted(person_key, face_results)
                    if not already_greeted and fr.get("quality", 0) >= 0.3:
                        self.vision_events.put({
                            "type": "unknown_visitor",
                            "identity": "unknown",
                            "timestamp": time.time(),
                            "face_bbox": fr["face_bbox"],
                            "score": fr["distance"],
                        })

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

    def _check_greeted(self, identity: str, face_results: list) -> bool:
        recent = list(self.vision_events.queue)[-5:]
        for evt in reversed(recent):
            if evt.get("type") == "unknown_visitor" and evt.get("identity") == identity:
                if time.time() - evt.get("timestamp", 0) < 15:
                    return True
        return False

    def _conversation_loop(self):
        """Consumes vision events, drives conversation via voice + LLM.

        Uses ConversationManager for state machine + agent.think/think_stream
        for LLM reasoning. Vision events are event-driven (not per-frame).
        """
        conv = ConversationManager(
            self.agent, self.voice, self.face_engine,
            vision_queue=self.action_events,
            frame_provider=lambda: (
                self._latest_frame.copy() if self._latest_frame is not None else None
            ),
        )

        while not self._stop.is_set():
            try:
                event = self.vision_events.get(timeout=0.5)
            except queue.Empty:
                continue

            if event["type"] == "unknown_visitor":
                face_data = {
                    "authorized": False,
                    "face_bbox": event["face_bbox"],
                    "score": event.get("score", 0),
                }
                # Greet: start TTS async, then listen (barge-in interrupts TTS)
                response = conv.start_for("unknown", face_data)
                if response:
                    self.voice.speak_async(response)
                    # Wait briefly for TTS to start, then listen (barge-in)
                    time.sleep(0.2)

                # Listen loop: capture speech, pass to ConversationManager
                while conv.state != "IDLE" and not self._stop.is_set():
                    transcript = self.voice.listen(timeout=10, phrase_limit=8)
                    if not transcript:
                        if conv.state != "ENROLLING":
                            conv.state = "IDLE"
                            break
                        continue

                    identity = conv.current_identity or "unknown"
                    response, action = conv.handle_response(
                        transcript, identity, face_data
                    )

                    if action == "enroll":
                        conv.state = "ENROLLING"
                        self._do_enrollment(conv, transcript)
                    elif action == "alert":
                        self.alerter.fire({
                            "reason": "unknown_visitor_refused_enrollment",
                            "transcript": transcript,
                        })
                        self.voice.speak_async("I'm escalating this to security.")
                        conv.reset()

            elif event["type"] == "frame_summary":
                self._status_text = (
                    f"Vision: {event['detections']} dets, "
                    f"{event['faces']} faces, "
                    f"{event['authorized']} authorized"
                )

    def _do_enrollment(self, conv: ConversationManager, name_hint: str):
        """Capture multiple face samples and enroll the visitor."""
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            conv.state = "IDLE"
            return

        faces = self.face_engine.detect(frame)
        if len(faces) == 0:
            self.voice.speak("I didn't catch a clear face. Please face the camera.")
            return

        name = name_hint.strip().split()[0] if name_hint else f"visitor_{int(time.time())}"

        # Multi-template enrollment: capture 3 embeddings for robustness
        templates = 0
        for _ in range(3):
            emb = self.face_engine.embed(frame, faces[0])
            if emb is not None and len(emb) > 0:
                self.face_engine.add_template(name, emb)
                templates += 1

        self.reasoner.set_authorized(set(self.face_engine.known_faces.keys()))

        mem = self.agent._memory_for(name)
        mem.persona.name = name
        mem.persona.relationship_tier = "trusted"
        mem.persona.purpose = "enrolled visitor"
        mem.persona.tags.append("enrolled")
        mem.add_interaction("system", f"Enrolled as {name} with {templates} templates")
        mem.save()

        self.memory[name] = mem
        if name in self.agent.memory:
            del self.agent.memory[name]

        conv.reset()
        self.voice.speak(f"Pleased to meet you, {name}! You're now authorized with {templates} face samples.")

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
