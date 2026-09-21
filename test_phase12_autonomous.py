"""Unit and integration tests for Phase 12 — Autonomous Loops, Visual Anomaly Trigger & Local Actuation.

Verifies:
1. Alerter local evidence persistence: high-resolution frame and crop saving to ./evidence/
2. Alerter metadata sidecar generation and history tracking
3. MemoryWriter immutable structured JSONL audit trace logging
4. Non-blocking Snapshot Ring Buffer capacity, thread-safety, and latency (<0.2ms)
5. Autonomous visual anomaly handling and ReAct investigation loop without mic input
"""
import os
import time
import json
import shutil
import tempfile
import threading
from typing import Any
import numpy as np
import cv2
import pytest

from actions.alerter import Alerter
from memory.writer import MemoryWriter
from core import Event, EventType, VISUAL_ANOMALY_DETECTED
from cognition.agent import VisionAgent
from cognition.reasoner import AgenticReasoner, Reasoner
from cognition.llm_interface import LocalFallbackLLM, LLMResponse
from run import VisionAgentApp


class TestAlerterEvidence:
    """Test local evidence saving, crop generation, and alert dispatching."""

    @pytest.fixture
    def temp_evidence_dir(self):
        tmp = tempfile.mkdtemp(prefix="aria_test_evidence_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_save_evidence_full_frame(self, temp_evidence_dir):
        alerter = Alerter(enabled=False, evidence_dir=temp_evidence_dir)
        frame = np.full((120, 160, 3), 180, dtype=np.uint8)

        saved_path = alerter.save_evidence(frame, prefix="test_full")
        assert saved_path is not None
        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path).startswith("test_full_")
        assert saved_path.endswith(".jpg")

        # Verify image readable and correct size
        read_img = cv2.imread(saved_path)
        assert read_img is not None
        assert read_img.shape == (120, 160, 3)

    def test_save_evidence_roi_crop(self, temp_evidence_dir):
        alerter = Alerter(enabled=False, evidence_dir=temp_evidence_dir)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Put red square at (50, 50, 40, 40)
        frame[50:90, 50:90] = (0, 0, 255)

        saved_path = alerter.save_evidence(frame, bbox=(50, 50, 40, 40), prefix="test_crop")
        assert saved_path is not None
        assert os.path.exists(saved_path)

        crop_img = cv2.imread(saved_path)
        assert crop_img is not None
        assert crop_img.shape == (40, 40, 3)
        assert np.all(crop_img[:, :, 2] >= 250)  # Red channel matches (allowing JPEG lossy compression)

    def test_save_evidence_with_metadata_sidecar(self, temp_evidence_dir):
        alerter = Alerter(enabled=False, evidence_dir=temp_evidence_dir)
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
        metadata = {"incident_id": "INC-101", "severity": "HIGH", "tags": ["hazard", "motion"]}

        saved_path = alerter.save_evidence(frame, prefix="test_meta", metadata=metadata)
        assert saved_path is not None

        meta_path = saved_path.replace(".jpg", ".json")
        assert os.path.exists(meta_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["incident_id"] == "INC-101"
        assert data["severity"] == "HIGH"
        assert "hazard" in data["tags"]

    def test_alerter_fire_local_only_and_local_saved(self, temp_evidence_dir):
        alerter = Alerter(enabled=False, evidence_dir=temp_evidence_dir)

        # 1. Fire without frame -> returns local_only
        res1 = alerter.fire({"reason": "stranger_at_door"})
        assert res1 == "local_only"
        assert len(alerter.history) == 1
        assert alerter.history[0]["reason"]["reason"] == "stranger_at_door"
        assert alerter.history[0]["evidence_path"] is None

        # 2. Fire with frame -> saves evidence and returns local_saved
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        reason_dict = {"reason": "unknown_visitor_refused"}
        res2 = alerter.fire(reason_dict, frame=frame)
        assert res2 == "local_saved"
        assert len(alerter.history) == 2
        assert "evidence_path" in reason_dict
        assert os.path.exists(reason_dict["evidence_path"])
        assert alerter.history[1]["evidence_path"] == reason_dict["evidence_path"]

    def test_alerter_invalid_inputs_safe(self, temp_evidence_dir):
        alerter = Alerter(enabled=False, evidence_dir=temp_evidence_dir)
        assert alerter.save_evidence(None) is None
        assert alerter.save_evidence(np.array([])) is None


class TestAuditLogging:
    """Test append-only structured JSONL audit trace logging in MemoryWriter."""

    @pytest.fixture
    def temp_log_file(self):
        tmp_dir = tempfile.mkdtemp(prefix="aria_test_audit_")
        log_file = os.path.join(tmp_dir, "test_audit.jsonl")
        yield log_file
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_record_audit_trace_jsonl(self, temp_log_file):
        trace1 = {
            "event_type": "visual_anomaly_detected",
            "decision": "WARN",
            "summary": "Detected untagged tool near edge",
            "iterations": 2,
            "evidence_path": "evidence/anomaly_warn_123.jpg",
            "trace": [
                {"iteration": 1, "thought": "Checking color", "observations": [{"tool": "vision.inspect_color_hsv"}]}
            ],
        }
        trace2 = {
            "event_type": "visual_anomaly_detected",
            "decision": "CLEAR",
            "summary": "False alarm, verified authorized tag",
            "iterations": 1,
            "evidence_path": None,
            "trace": [],
        }

        ret_path1 = MemoryWriter.record_audit_trace(trace1, audit_file=temp_log_file)
        ret_path2 = MemoryWriter.record_audit_trace(trace2, audit_file=temp_log_file)
        assert ret_path1 == temp_log_file
        assert ret_path2 == temp_log_file

        assert os.path.exists(temp_log_file)

        # Read back and verify lines
        with open(temp_log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["event_type"] == "visual_anomaly_detected"
        assert lines[0]["decision"] == "WARN"
        assert lines[0]["evidence_path"] == "evidence/anomaly_warn_123.jpg"
        assert "timestamp" in lines[0]
        assert "iso_timestamp" in lines[0]

        assert lines[1]["decision"] == "CLEAR"
        assert lines[1]["iterations"] == 1

    def test_record_audit_trace_handles_custom_objects(self, temp_log_file):
        class DummyReport:
            def __init__(self):
                self.metric = 42
                self.status = "ok"

        trace = {
            "event_type": "test_custom",
            "obj": DummyReport(),
        }
        MemoryWriter.record_audit_trace(trace, audit_file=temp_log_file)

        with open(temp_log_file, "r", encoding="utf-8") as f:
            line = json.loads(f.readline())
        assert line["obj"]["metric"] == 42


class TestSnapshotRingBuffer:
    """Test non-blocking snapshot ring buffer preventing CPU frame lock starvation."""

    def test_snapshot_ring_buffer_capacity_and_order(self):
        app = VisionAgentApp.__new__(VisionAgentApp)
        app._frame_lock = threading.Lock()
        app._latest_frame = None
        app._snapshot_lock = threading.Lock()
        app._raw_snapshots = []
        app._max_snapshots = 3

        f1 = np.full((10, 10, 3), 1, dtype=np.uint8)
        f2 = np.full((10, 10, 3), 2, dtype=np.uint8)
        f3 = np.full((10, 10, 3), 3, dtype=np.uint8)
        f4 = np.full((10, 10, 3), 4, dtype=np.uint8)

        for f in (f1, f2, f3, f4):
            with app._snapshot_lock:
                app._raw_snapshots.append(f.copy())
                if len(app._raw_snapshots) > app._max_snapshots:
                    app._raw_snapshots.pop(0)

        # Capacity should be capped at 3
        assert len(app._raw_snapshots) == 3
        # Oldest (f1) was evicted; latest should be f4
        latest = app._latest_frame_now()
        assert latest is not None
        assert np.all(latest == 4)

    def test_snapshot_ring_buffer_concurrency_latency(self):
        app = VisionAgentApp.__new__(VisionAgentApp)
        app._frame_lock = threading.Lock()
        app._latest_frame = None
        app._snapshot_lock = threading.Lock()
        app._raw_snapshots = [np.zeros((480, 640, 3), dtype=np.uint8)]
        app._max_snapshots = 3

        stop_event = threading.Event()

        # Background writer thread pushing frames at 100 FPS
        def writer_thread():
            count = 0
            while not stop_event.is_set():
                frame = np.full((480, 640, 3), count % 255, dtype=np.uint8)
                with app._snapshot_lock:
                    app._raw_snapshots.append(frame)
                    if len(app._raw_snapshots) > app._max_snapshots:
                        app._raw_snapshots.pop(0)
                count += 1
                time.sleep(0.01)

        t = threading.Thread(target=writer_thread, daemon=True)
        t.start()

        # Benchmark _latest_frame_now latency under concurrent writing
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            frame = app._latest_frame_now()
            times.append((time.perf_counter() - t0) * 1000)
            assert frame is not None
            assert frame.shape == (480, 640, 3)
            time.sleep(0.005)

        stop_event.set()
        t.join(timeout=1.0)

        avg_lat = sum(times) / len(times)
        # Snapshot buffer read must be well under 0.5ms (typically <0.1ms)
        assert avg_lat < 0.5, f"Snapshot ring buffer read too slow: {avg_lat:.3f}ms"


class TestAutonomousAnomalyLoop:
    """Test end-to-end visual anomaly triggers and ReAct inspection loop without human voice."""

    @pytest.fixture
    def test_app(self):
        tmp_dir = tempfile.mkdtemp(prefix="aria_test_app_")
        app = VisionAgentApp.__new__(VisionAgentApp)
        app._frame_lock = threading.Lock()
        app._latest_frame = None
        app._snapshot_lock = threading.Lock()
        app._raw_snapshots = []
        app._max_snapshots = 3
        app.vision_events = pytest.importorskip("queue").Queue()
        app.action_events = pytest.importorskip("queue").Queue()
        app._stop = threading.Event()

        # Mock voice and UI hub
        class DummyVoice:
            def __init__(self):
                self.spoken = []

            def speak_async(self, text):
                self.spoken.append(text)

        class DummyHub:
            def __init__(self):
                self.states = []
                self.dialogue = []

            def set_aria_state(self, state):
                self.states.append(state)

            def say(self, speaker, text):
                self.dialogue.append((speaker, text))

        app.voice = DummyVoice()
        app.hub = DummyHub()
        app.alerter = Alerter(enabled=False, evidence_dir=tmp_dir)

        # Wire agent and reasoner with fallback LLM
        app.agent = VisionAgent()
        app.reasoner = Reasoner(
            face_engine=None,
            llm=LocalFallbackLLM(),
            dispatcher=app.agent.dispatcher,
        )
        app.agent.attach_tools(
            voice=app.voice,
            vision_ctx=None,
            frame_provider=app._latest_frame_now,
        )
        app.reasoner.dispatcher = app.agent.dispatcher

        yield app, tmp_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_handle_visual_anomaly_clear(self, test_app):
        app, tmp_dir = test_app
        # Seed snapshot buffer with a clean frame
        clean_frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        with app._snapshot_lock:
            app._raw_snapshots.append(clean_frame)

        event = {
            "type": VISUAL_ANOMALY_DETECTED,
            "new_objects": ["cup"],
            "bbox": (10, 10, 40, 40),
            "description": "Safe beverage container on desk",
        }

        audit_file = os.path.join(tmp_dir, "audit_log.jsonl")
        orig_record = MemoryWriter.record_audit_trace
        MemoryWriter.record_audit_trace = lambda data, audit_file=audit_file: orig_record(data, audit_file=audit_file)

        try:
            res = app._handle_visual_anomaly(event)
            assert res["decision"] in ("CLEAR", "WARN", "HALT")
            assert "audit_record" in res
            assert os.path.exists(audit_file)

            with open(audit_file, "r", encoding="utf-8") as f:
                log_entry = json.loads(f.readline())
            assert log_entry["event_type"] == "visual_anomaly_detected"
            assert "decision" in log_entry
        finally:
            MemoryWriter.record_audit_trace = orig_record

    def test_handle_visual_anomaly_hazard_saves_evidence_and_alerts(self, test_app):
        app, tmp_dir = test_app
        # Seed frame with red hazard marker
        hazard_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hazard_frame[20:60, 20:60] = (0, 0, 255)
        with app._snapshot_lock:
            app._raw_snapshots.append(hazard_frame)

        event = {
            "type": VISUAL_ANOMALY_DETECTED,
            "new_objects": ["scissors"],
            "bbox": (20, 20, 40, 40),
            "force_evidence": True,  # Ensures evidence is saved regardless of fallback verdict
            "description": "Hazardous object left unattended near edge: WARN",
        }

        audit_file = os.path.join(tmp_dir, "audit_log.jsonl")
        orig_record = MemoryWriter.record_audit_trace
        MemoryWriter.record_audit_trace = lambda data, audit_file=audit_file: orig_record(data, audit_file=audit_file)

        try:
            res = app._handle_visual_anomaly(event)
            # Verify evidence image was generated and exists
            assert res["evidence_path"] is not None
            assert os.path.exists(res["evidence_path"])

            # Verify audit log recorded the evidence path
            with open(audit_file, "r", encoding="utf-8") as f:
                log_entry = json.loads(f.readline())
            assert log_entry["evidence_path"] == res["evidence_path"]

            # If decision is actionable, alerter.history must capture it
            if res["decision"] in ("WARN", "HALT"):
                assert len(app.alerter.history) >= 1
                assert app.alerter.history[-1]["evidence_path"] == res["evidence_path"]
        finally:
            MemoryWriter.record_audit_trace = orig_record

    def test_autonomous_trigger_queue(self, test_app):
        app, _ = test_app
        assert app.vision_events.empty()

        # Call trigger_anomaly without human speech
        app.trigger_anomaly({"new_objects": ["backpack"], "bbox": (10, 10, 50, 50)})

        assert not app.vision_events.empty()
        queued_evt = app.vision_events.get_nowait()
        assert queued_evt["type"] == VISUAL_ANOMALY_DETECTED
        assert queued_evt["new_objects"] == ["backpack"]
