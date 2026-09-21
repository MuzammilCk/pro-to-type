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
    _dotenv = {}
    with open(_env_path) as _f:
        for line in _f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                _dotenv[k.strip()] = v.strip()
    for k, v in _dotenv.items():
        os.environ.setdefault(k, v)

import numpy as np

from detector import Detector
from reasoner import Reasoner
from alerter import Alerter
from face_engine import FaceEngine
from tracker import FaceTracker
from presence import PresenceManager
from input_source import WebcamSource
from conversation import ConversationManager
from agent import VisionAgent, VisionContext, VoiceSession
from voice import SarvamVoice
from mic_vad import MicVAD
from companion import ProactiveEngine
from context_memory import PersonMemory, SessionManager
from webui import UiHub, serve, render_jpeg
from events import Event, EventType


CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "models/yolov5s.onnx"),
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

        # Talkative upgrade (GPT-Live-style): voice session pieces. Created
        # here, started in the conversation thread.
        self.mic: MicVAD | None = None
        self.vision_ctx: VisionContext | None = None
        self.session: VoiceSession | None = None
        self.proactive: ProactiveEngine | None = None

        # Thread-safe event queues
        self.vision_events: queue.Queue = queue.Queue()
        self.action_events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._farewells_spoken: set[str] = set()
        self._unknown_greeted: bool = False

        # Restore session state
        saved = SessionManager.load_current_state()
        if saved:
            self.memory.update(saved)
            print(f"[Session] Resumed {len(saved)} person memory(s)")

        # Input: webcam only (owner decision — IPCam/Phone/File inputs removed;
        # dev phase has exactly one camera and one person on it).
        self.source = WebcamSource(0)

        # Web dashboard (replaces the cv2.imshow debug window)
        self.hub = UiHub()

        # Latest frame + analysis (shared between vision thread and render thread)
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_decision = None
        self._status_text = ""

        # Face tracking + presence management (architecture: separate seeing from thinking)
        self.tracker = FaceTracker(self.face_engine)
        self.presence = PresenceManager(self.vision_events)

        # Face-state overlay moved from cv2 pixels to DOM (web UI): the
        # stranger-scan indicator is now drawn by the browser. This also
        # removes the MediaPipe mesh render from the hot loop — the same
        # CPU reasoning as the YOLO throttle: conversation > visualization.
        # Companion-phase CPU budget (owner decision, dev phase: owner is the
        # only person on camera): YOLO is visualization/awareness only, so it
        # runs on a slow TIME-based cadence instead of every frame. At 640x640
        # it was ~150-250ms/frame — the single biggest CPU cost (3.9 FPS), and
        # that starvation delayed the VAD mic and speech threads. Set
        # ARIA_YOLO_EVERY_SEC=0 to disable YOLO entirely.
        self._yolo_every = float(os.getenv("ARIA_YOLO_EVERY_SEC", "2.0"))
        self._last_yolo = 0.0
        self._last_detections = None

    def run(self):
        httpd, _ui_t = serve(self.hub)
        port = httpd.server_address[1]
        print(f"ARIA UI:  http://127.0.0.1:{port}   (Ctrl+C in this terminal to quit)")

        vision_t = threading.Thread(target=self._vision_loop, daemon=True)
        voice_t = threading.Thread(target=self._conversation_loop, daemon=True)
        vision_t.start()
        voice_t.start()

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()

        vision_t.join(timeout=2)
        voice_t.join(timeout=2)
        if self.mic is not None:
            self.mic.stop()
        if self.proactive is not None:
            self.proactive.stop()
        self.source.close()
        httpd.shutdown()

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
            ret, frame = self.source.read()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            t0 = time.perf_counter()

            # YOLO object detection — time-throttled (companion-phase budget).
            # Face detection/recognition still runs EVERY frame; YOLO boxes
            # are just HUD/awareness and refresh every _yolo_every seconds.
            if self._yolo_every > 0 and (t0 - self._last_yolo) >= self._yolo_every:
                detections = self.detector.detect(frame)
                self._last_detections = detections
                self._last_yolo = t0
            else:
                detections = self._last_detections

            # Face tracking (with frame-skipping + downscale)
            face_results = self.tracker.update(frame, downscale_factor=0.5)

            # Feed "what ARIA sees" to the conversational brain (Phase 3)
            if self.vision_ctx is not None:
                self.vision_ctx.update(face_results, detections)

            # Run recognition only on faces that need it (new or stale tracks)
            if any(fr.get("needs_recognition") for fr in face_results):
                self.tracker.recognize_faces(frame)
                face_results = self.tracker.update_results()

            # Presence: emit de-duplicated events (person_entered, person_recognized, etc.)
            self.presence.update(face_results)

            # Clean up lost faces
            self.tracker.cleanup_left_faces()

            # UI state: normalized face boxes (browser draws the overlays)
            ih, iw = frame.shape[:2]
            faces_ui = []
            known_names = []
            strangers = 0
            for fr in face_results:
                x, y, w, h = fr["face_bbox"]
                is_known = bool(fr["authorized"])
                if is_known:
                    known_names.append(fr.get("identity") or "known")
                else:
                    strangers += 1
                faces_ui.append({
                    "kind": "known" if is_known else "stranger",
                    "name": fr.get("identity") or "",
                    # tracker exports match confidence as "distance" (cosine
                    # similarity — higher is better); same mapping presence.py uses
                    "score": fr.get("distance", 0.0),
                    "x": x / iw, "y": y / ih, "w": w / iw, "h": h / ih,
                })
            self.hub.set_faces(faces_ui)
            self.hub.set_people(
                [{"kind": "known", "name": n} for n in known_names]
                + ([{"kind": "scan", "count": strangers}] if strangers else []))
            jpeg = render_jpeg(frame)
            if jpeg is not None:
                self.hub.update_frame(jpeg)

            fps_buf.append(time.perf_counter() - t0)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            avg_dt = sum(fps_buf) / len(fps_buf) if fps_buf else 0.1
            yolo_state = ("off" if self._yolo_every <= 0
                          else f"{self._yolo_every:.0f}s")
            self._status_text = (f"{1/avg_dt:.1f} FPS | faces={len(face_results)} | "
                                 f"tracks={self.tracker.track_count} | yolo={yolo_state}")
            self.hub.set_status({"fps": 1 / avg_dt if avg_dt else 0.0,
                                 "yolo": yolo_state})

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_decision = {
                    "face_results": face_results,
                    "should_act": any(not fr["authorized"] for fr in face_results),
                    "total": len(detections or []),
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
        """Enroll the visitor under `name` via the Phase-2 enrollment API.

        Captures INDEPENDENT samples from FRESH frames (person moves between
        captures) and hands them to face_engine.enroll_identity(), which
        validates quality, duplicates, and coherence centrally. The old
        "one frame, N embeddings" shortcut can no longer enroll.
        """
        samples = []
        for attempt in range(8):  # ~a few seconds of live capture
            frame, face = self._latest_stable_face()
            if frame is not None and face is not None:
                samples.append((frame, face))
                time.sleep(0.6)  # natural movement between captures
            if len(samples) >= 5:
                break

        res = self.face_engine.enroll_identity(name, samples)
        if not res["enrolled"]:
            self._speak("I couldn't get enough clear, distinct looks at your face. "
                        "Let's try that again in better light.")
            print(f"[Enroll] rejected: {res['reason']} ({res['accepted']}/{res['n_samples']} accepted)")
            conv.reset()
            return

        self.reasoner.set_authorized(set(self.face_engine.known_faces.keys()))

        mem = self.agent._memory_for(name)
        mem.persona.name = name
        mem.persona.relationship_tier = "trusted"
        mem.persona.purpose = face_data.get("purpose") or "enrolled visitor"
        mem.persona.tags.append("enrolled")
        mem.add_interaction("system", f"Enrolled as {name} with {res['templates']} validated templates")
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

        # Voice session stack (talkative upgrade): VAD mic, split-brain
        # session, vision context, proactive companion.
        self.mic = MicVAD()
        if not self.mic.start():
            print("[ARIA] WARNING: microphone unavailable — voice input will not work this session.")
        # Echo guard (pseudo-duplex): while ARIA speaks, the VAD suppresses
        # input so she never transcribes her own voice from the speakers.
        # The same signal drives the UI orb (speaking <-> listening).
        def _echo_guard(speaking: bool):
            self.mic.speaking = speaking
            self.hub.set_aria_state("speaking" if speaking else "listening")

        self.voice.set_speaking_callback(_echo_guard)
        # Web UI transcript: every spoken line and heard turn lands in the rail
        self.voice.set_ui_hooks(on_say=self.hub.say, on_heard=self.hub.heard)
        # Phase 2: wire voice events (USER_UTTERANCE, AGENT_*_SPEAKING) into action_events queue
        self.voice.set_event_queue(self.action_events)
        self.vision_ctx = VisionContext()
        self.session = VoiceSession(self.agent, self.voice, mic=self.mic,
                                    vision=self.vision_ctx)
        self.proactive = ProactiveEngine(self.agent, self.voice, self.vision_ctx,
                                         session=self.session)
        self.proactive.start()

        while not self._stop.is_set():
            try:
                event = self.vision_events.get(timeout=0.5)
            except queue.Empty:
                continue

            evt_type = event.get("type", "")

            if evt_type == EventType.PERSON_ENTERED:
                # Arrival noted; follow-up IDENTITY_CONFIRMED or person_unrecognized handles greeting
                pass

            elif evt_type == EventType.IDENTITY_CONFIRMED:
                # B5 fix: greet known people, then LISTEN — they get a dialogue,
                # not a monologue followed by silence.
                name = event.get("name", "friend")
                self._farewells_spoken.discard(name)
                track_id = event.get("track_id", "")
                face_data = {
                    "authorized": True,
                    "face_bbox": event.get("face_bbox", (0, 0, 0, 0)),
                    "score": event.get("score", 0.0),
                }
                self.presence.mark_greeted(track_id)

                # Phase 3: let the conversational brain know who showed up
                if self.vision_ctx is not None and name not in ("", "friend", "unknown"):
                    self.vision_ctx.last_known_name = name
                    self.vision_ctx.add_event(f"{name} arrived and was recognized.")

                response, action = self.agent.think(name, face_data)
                self._speak(response)
                self._dialogue_loop(conv, name, face_data)

            elif evt_type == "person_unrecognized":
                # B1 fix: the greeting is built once and spoken once by the
                # caller (start_for returns text; playback happens here).
                prev = event.get("previous_identity")
                track_id = event.get("track_id", "")

                # Phase 8 — revocation propagation. When the tracker revokes
                # an identity with a live conversation, the conversation MUST
                # NOT continue addressing that person by the old name.
                # Terminate the authenticated dialogue first; fresh face
                # evidence is the only path back to a named conversation.
                if prev and prev not in ("", "unknown") and conv.current_identity == prev:
                    print(f"[Security] '{prev}' lost authorization mid-conversation — "
                          "suspending authenticated dialogue.")
                    conv.reset()
                    if self.vision_ctx is not None:
                        self.vision_ctx.clear_identity(prev)
                        self.vision_ctx.add_event(
                            f"The person who was {prev} is no longer recognized.")
                    self._speak("Hold on — I've lost track of who you are. "
                                "Let me take a fresh look.")

                # Cold-start / entry grace window: give face recognition up to 350ms to verify
                # a freshly appeared track before assuming it is an unknown stranger.
                track = self.presence.tracks.get(track_id)
                if track and not track.authorized and (time.time() - track.first_seen < 0.35):
                    rem = 0.35 - (time.time() - track.first_seen)
                    if rem > 0:
                        time.sleep(rem)

                # Gate 1: If settled as authorized, or if an authorized person is present, skip unknown greeting
                if (track and track.authorized) or any(t.get("authorized") for t in self.tracker.tracks.values()):
                    print(f"[Presence] Skipping unknown greeting for {track_id!r} "
                          "(authorized person currently present)")
                    continue

                # Gate 2: skip the greeting entirely for phantom churn tracks.
                # should_greet() checks both the track-level cooldown AND the
                # identity-level cooldown — so a phantom track that spawned
                # right after another unknown was just greeted will be
                # suppressed without entering _dialogue_loop at all.
                if not self.presence.should_greet(track_id):
                    print(f"[Presence] Skipping duplicate greeting for {track_id!r} "
                          "(already greeted recently)")
                    continue

                # Only clear unknown-farewell suppression once a real greeting is actually spoken
                self._farewells_spoken.discard("unknown")
                face_data = {
                    "authorized": False,
                    "face_bbox": event.get("face_bbox", (0, 0, 0, 0)),
                    "score": event.get("score", 0.0),
                }
                self.presence.mark_greeted(track_id)

                if self.vision_ctx is not None:
                    self.vision_ctx.add_event("A new visitor arrived and is being scanned.")

                greeting = conv.start_for("unknown", face_data)
                self._speak(greeting)
                self._unknown_greeted = True
                self._dialogue_loop(conv, "unknown", face_data)

            elif evt_type == EventType.PERSON_LEFT:
                self._handle_person_left(event, conv)

            elif evt_type == "frame_summary":
                self._status_text = (
                    f"Vision: {event['detections']} dets, "
                    f"{event['faces']} faces, "
                    f"{event['authorized']} authorized"
                )

    def _handle_person_left(self, event: dict, conv) -> None:
        name = event.get("name") or event.get("identity", "")
        # Prefer the persona display name stored in memory over the raw
        # identity key (which may be "MuzammilCK" vs "Muzammil").
        try:
            mem = self.agent._memory_for(name)
            display = (mem.persona.name or name) if mem else name
        except Exception:
            display = name
        farewell = (
            f"See you later, {display}. Take care!"
            if display and display not in ("", "unknown")
            else "Goodbye! Come back anytime."
        )
        conv.reset()
        if self.mic is not None:
            self.mic.clear_pending()
        key = name if name and name not in ("", "unknown") else "unknown"
        # Track churn guard: if the identity is still authorized on a
        # *different* live track (e.g. face_0 expired but face_12 is
        # already confirmed as the same person), the person hasn't
        # actually left — suppress the farewell and leave the hub state
        # alone so the active dialogue continues uninterrupted.
        if name and name not in ("", "unknown") and self._identity_currently_authorized(name):
            pass  # person still present on another track — no farewell
        elif key == "unknown" and (not getattr(self, "_unknown_greeted", False) or any(t.get("authorized") for t in self.tracker.tracks.values())):
            pass  # phantom track expired while an authorized person is present (or no stranger was ever greeted) — no farewell
        elif key in self._farewells_spoken:
            pass  # farewell already spoken for this identity since last confirmation
        else:
            self._farewells_spoken.add(key)
            if key == "unknown":
                self._unknown_greeted = False
            self.hub.set_aria_state("idle")
            self.voice.speak_async(farewell)

    def _identity_currently_authorized(self, identity: str) -> bool:
        """True iff some live, authorized tracker track currently carries this
        identity (Phase 8: conversation identity = live face evidence)."""
        for track in self.tracker.tracks.values():
            if track.get("authorized") and track.get("identity") == identity:
                return True
        return False

    def _dialogue_loop(self, conv, identity, face_data):
        """Shared listen/respond loop for recognized AND unknown visitors.

        B5 fix: recognized visitors previously never reached a listen() call,
        so anything they said after the greeting fell into the void.

        Talkative upgrade (GPT-Live-style):
        - VAD mic: capture starts at speech onset, ends at natural pause —
          no fixed 8s windows.
        - CRITICAL MUTE FIX: dialogue replies are now actually SPOKEN. The
          old loop called conv.handle_response() and discarded the text —
          ARIA went silent after the greeting.
        - Normal chat turns go through the split-brain VoiceSession (short
          spoken answers, delegation to the backend brain for depth).
        - Session stays open through idle gaps; a gentle nudge after ~90s.
        """
        if conv.current_identity is None:
            conv.current_identity = identity
        if self.session is not None:
            self.session.touch_activity()
        # Phase 8 synchronous guard: a named dialogue may only continue while
        # the CURRENT tracker state still authorizes that identity. This
        # catches revocation even while the conversation thread is blocked in
        # listen() (the event path only fires between turns).
        if identity not in ("", "unknown") and not self._identity_currently_authorized(identity):
            print(f"[Security] '{identity}' is no longer authorized — "
                  "closing authenticated dialogue.")
            conv.reset()
            self.hub.set_aria_state("idle")
            return

        idle_turns = 0
        while conv.state != "IDLE" and not self._stop.is_set():
            # Prompt departure check: exit immediately if vision confirms room is empty.
            # Do NOT speak a farewell here — the PERSON_LEFT event that caused this
            # condition is already queued in vision_events and will be dequeued by
            # conversation_thread as soon as this method returns, producing the
            # farewell exactly once via the PERSON_LEFT handler.
            if self.vision_ctx is not None and "No one is in view" in self.vision_ctx.context_text():
                conv.reset()
                self.hub.set_aria_state("idle")
                break

            timeout = 12.0 if self.mic is not None else 12.0
            self.hub.set_aria_state("listening")
            _t_listen_entry = time.time()
            transcript = self.voice.listen(timeout=timeout, phrase_limit=8,
                                           mic=self.mic)
            _t_listen_exit = time.time()
            if transcript:
                print(f"[Timing] listen() total wall time: {_t_listen_exit - _t_listen_entry:.3f}s")
            if not transcript:
                # Re-check presence right after timeout before burning another idle turn
                if self.vision_ctx is not None and "No one is in view" in self.vision_ctx.context_text():
                    conv.reset()
                    self.hub.set_aria_state("idle")
                    break
                idle_turns += 1
                if conv.state == "ENROLLING":
                    continue
                # Long silence: one gentle nudge, then close politely
                nudged = (self.session is not None
                          and self.session.maybe_nudge(identity, face_data))
                if not nudged and idle_turns >= 3:
                    conv.reset()
                    break
                continue

            idle_turns = 0
            if self.session is not None:
                self.session.touch_activity()
            _t_think_start = time.time()
            self.hub.set_aria_state("thinking")

            response, action = conv.handle_response(transcript, conv.current_identity, face_data)
            _t_think_done = time.time()
            print(f"[Timing] LLM think: {_t_think_done - _t_think_start:.3f}s")

            if action == "enroll":
                conv.state = "ENROLLING"
                # B3 fix: derive the name from what they actually said; if we
                # can't, ask instead of enrolling "Hey" as a person.
                name = self._extract_name(transcript)
                if name is None:
                    self._speak("And what should I call you?")
                    name_reply = self.voice.listen(timeout=8, phrase_limit=5,
                                                   mic=self.mic)
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
            elif action == "escalate" or action == "recognized":
                # State-machine verdicts (turn limit / recognized handshake):
                # speak the composed response — never discard it again.
                self._speak(response)
                if action == "escalate":
                    conv.reset()
                    break
            else:
                # Normal chat: the split-brain session answers AND speaks
                # (short spoken turn; deep questions are delegated to the
                # backend brain with fillers while it works).
                if self.session is not None:
                    self.session.run_turn(identity, face_data, user_text=transcript)
                else:
                    self._speak(response)
            # loop continues — listen for their next line

    # ------------------------------------------------------------------




if __name__ == "__main__":
    app = VisionAgentApp()
    app.run()
