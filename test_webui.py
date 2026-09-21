"""Offline tests for the ARIA web UI (webui.py).

Run:  python test_webui.py   (or pytest test_webui.py -q)

Covers:
  UiHub — transcript ordering/dedup counters, aria-state validation,
          face/people/status snapshots, empty-say rejection
  render_jpeg — encodes a real numpy frame
  HTTP — GET / (UI page), GET /state (JSON), GET /stream (MJPEG frames)

No camera, no microphone, no network beyond 127.0.0.1.
"""
import json
import threading
import time
import urllib.request

import numpy as np

from interaction.webui import UiHub, render_jpeg, serve


def test_transcript_ordering_and_roles():
    hub = UiHub()
    hub.say("Hello, Muzammil!", role="aria")
    hub.heard("hi aria")
    hub.say("")  # empty lines must be ignored
    snap = hub.snapshot()
    roles = [l["role"] for l in snap["transcript"]]
    texts = [l["text"] for l in snap["transcript"]]
    assert roles == ["aria", "user"], roles
    assert texts == ["Hello, Muzammil!", "hi aria"], texts
    # monotonic ids drive the browser's incremental renderer
    ids = [l["i"] for l in snap["transcript"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_aria_state_validation():
    hub = UiHub()
    for valid in ("idle", "listening", "thinking", "speaking"):
        hub.set_aria_state(valid)
        assert hub.snapshot()["aria_state"] == valid
    hub.set_aria_state("bogus")
    assert hub.snapshot()["aria_state"] == "idle"


def test_faces_people_status_snapshot():
    hub = UiHub()
    hub.set_faces([{"kind": "known", "name": "Muzammil",
                    "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.3}])
    hub.set_people([{"kind": "known", "name": "Muzammil"}])
    hub.set_status({"fps": 12.5, "yolo": "2s"})
    snap = hub.snapshot()
    assert snap["faces"][0]["name"] == "Muzammil"
    assert snap["people"][0]["kind"] == "known"
    assert snap["status"]["fps"] == 12.5
    # snapshot returns copies — mutating them must not corrupt the hub
    snap["faces"][0]["name"] = "hacked"
    assert hub.snapshot()["faces"][0]["name"] == "Muzammil"


def test_render_jpeg_roundtrip():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # make it non-trivial
    jpeg = render_jpeg(frame)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_http_endpoints():
    hub = UiHub()
    hub.say("Testing 1 2 3", role="aria")
    hub.update_frame(render_jpeg(np.full((32, 32, 3), 90, dtype=np.uint8)))
    httpd, _ = serve(hub, port=0)  # ephemeral port
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        # / — the UI page
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
            assert "ARIA" in html and "/stream" in html and "/state" in html
        # /state — JSON with our transcript line
        with urllib.request.urlopen(base + "/state", timeout=5) as r:
            state = json.loads(r.read().decode("utf-8"))
            assert state["transcript"][-1]["text"] == "Testing 1 2 3"
            assert state["aria_state"] == "idle"
        # /stream — MJPEG: at least one complete JPEG part arrives
        parts = []
        with urllib.request.urlopen(base + "/stream", timeout=5) as r:
            deadline = time.time() + 5
            buf = b""
            while time.time() < deadline and parts.count(b"\xff\xd8") < 1:
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start)
                if start != -1 and end != -1:
                    parts.append(buf[start:end])
                    break
        assert parts, "no complete JPEG frame received from /stream"
    finally:
        httpd.shutdown()


def test_faces_ui_contract_score_key():
    """run.py builds faces_ui entries from tracker results — the browser HUD
    reads f.score (tracker's "distance" cosine similarity). If this contract
    regresses, known faces render as 0% again."""
    fr = {"identity": "Muzammil", "authorized": True, "distance": 0.62,
          "face_bbox": (40, 40, 80, 80), "landmarks": [], "quality": 0.8}
    iw, ih = 640, 480
    x, y, w, h = fr["face_bbox"]
    faces_ui = [{
        "kind": "known" if fr["authorized"] else "stranger",
        "name": fr.get("identity") or "",
        "score": fr.get("distance", 0.0),
        "x": x / iw, "y": y / ih, "w": w / iw, "h": h / ih,
    }]
    hub = UiHub()
    hub.set_faces(faces_ui)
    entry = hub.snapshot()["faces"][0]
    assert "score" in entry, "faces_ui entry lost the score key — HUD will show 0%"
    assert entry["score"] == 0.62
    # browser math: Math.round((f.score || 0) * 100) + "%"
    assert round((entry["score"] or 0) * 100) == 62


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
