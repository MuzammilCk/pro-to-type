"""ARIA web UI — local dashboard replacing the cv2.imshow debug window.

Stdlib-only (http.server + cv2 JPEG encoding): no new dependencies.

Architecture:
  - Vision thread (run.py) pushes JPEG frames + state snapshots into UiHub
  - GET /stream  — MJPEG video feed (pure vision frame, no cv2 scribbles)
  - GET /state   — JSON: faces (normalized boxes), people, transcript,
                   ARIA's conversational state (idle/listening/thinking/speaking)
  - GET /        — the UI: video stage with DOM overlays + companion rail

Design contract (ARIA_UI_DESIGN.md):
  - warm charcoal + amber = ARIA's voice; cyan is RESERVED for the
    stranger-scan state — the DOM corner brackets replace the old cv2 mesh
  - recognized people: nothing over their face but a soft name caption
  - transcript-first: every word ARIA hears or says lands in the rail
"""
import os
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class UiHub:
    """Thread-safe state + frame exchange between the vision/voice threads
    and the HTTP server."""

    def __init__(self, transcript_len: int = 60):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_seq = 0
        self._faces: list[dict] = []
        self._people: list[dict] = []
        self._status: dict = {}
        self._aria_state = "idle"
        self._transcript: deque = deque(maxlen=transcript_len)
        self._t_counter = 0

    # ---- producers (vision thread / voice hooks) -----------------------

    def update_frame(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg
            self._frame_seq += 1

    def set_faces(self, faces: list[dict]):
        """faces: [{kind, name, score, x, y, w, h}] — normalized 0..1."""
        with self._lock:
            self._faces = faces

    def set_people(self, people: list[dict]):
        with self._lock:
            self._people = people

    def set_status(self, status: dict):
        with self._lock:
            self._status = status

    def set_aria_state(self, state: str):
        state = state if state in ("idle", "listening", "thinking", "speaking") else "idle"
        with self._lock:
            self._aria_state = state

    def say(self, text: str, role: str = "aria"):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._t_counter += 1
            self._transcript.append({
                "i": self._t_counter,
                "role": role,
                "text": text,
                "t": time.strftime("%H:%M"),
            })

    def heard(self, text: str):
        self.say(text, role="user")

    # ---- consumer (HTTP handlers) --------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "aria_state": self._aria_state,
                "status": dict(self._status),
                "faces": [dict(f) for f in self._faces],
                "people": [dict(p) for p in self._people],
                "transcript": list(self._transcript),
            }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg


def render_jpeg(frame, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf) if ok else None


class _Handler(BaseHTTPRequestHandler):

    hub: UiHub  # set by serve()

    def log_message(self, *args):  # silence request spam
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/state":
            body = json.dumps(self.hub.snapshot()).encode("utf-8")
            self._send(200, body, "application/json")
        elif self.path == "/stream":
            self._stream_mjpeg()
        else:
            self._send(404, b"not found", "text/plain")

    def _stream_mjpeg(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=ariaframe")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_seq = -1
            stale_repeats = 0
            while True:
                jpeg = self.hub.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.2)
                    continue
                with self.hub._lock:
                    seq = self.hub._frame_seq
                if seq == last_seq:
                    stale_repeats += 1
                    if stale_repeats > 30:  # ~3s without a new frame -> close
                        break
                    time.sleep(0.1)
                    continue
                last_seq, stale_repeats = seq, 0
                self.wfile.write(
                    b"--ariaframe\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n")
                time.sleep(0.033)  # ~30 fps ceiling; vision loop sets real pace
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARIA</title>
<style>
  :root {
    --bg: #141210;         /* warm charcoal */
    --panel: #1e1b17;
    --line: #35302a;
    --amber: #f2a33c;      /* ARIA's voice */
    --amber-soft: #ffd9a0;
    --cyan: #53d6e0;       /* scanning ONLY */
    --cream: #ede6da;
    --dim: #9a8f80;
    --display: "Bahnschrift", "Segoe UI", sans-serif;
    --body: "Segoe UI", system-ui, sans-serif;
    --mono: "Cascadia Mono", Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg); color: var(--cream);
    font-family: var(--body); display: flex; flex-direction: column;
  }
  header {
    display: flex; align-items: baseline; gap: 14px;
    padding: 14px 22px 10px; border-bottom: 1px solid var(--line);
  }
  header h1 {
    font-family: var(--display); font-weight: 600; font-size: 20px;
    letter-spacing: 0.35em; text-transform: uppercase;
  }
  header .sub { color: var(--dim); font-size: 12.5px; }
  header .status {
    margin-left: auto; font-family: var(--mono); font-size: 12px;
    color: var(--dim);
  }
  main { flex: 1; display: flex; min-height: 0; }

  /* ---- video stage ---- */
  .stage-wrap { flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 20px; min-width: 0; }
  .stage { position: relative; max-width: 100%; max-height: 100%; }
  .stage img {
    display: block; max-width: 100%; max-height: calc(100vh - 130px);
    border-radius: 10px; border: 1px solid var(--line);
  }
  .facebox { position: absolute; pointer-events: none; }
  .facebox.stranger::before, .facebox.stranger::after,
  .facebox.stranger .c1, .facebox.stranger .c2 {
    content: ""; position: absolute; width: 14px; height: 14px;
    border: 2px solid var(--cyan);
  }
  .facebox.stranger::before { top: 0; left: 0; border-right: none; border-bottom: none; }
  .facebox.stranger::after  { top: 0; right: 0; border-left: none; border-bottom: none; }
  .facebox.stranger .c1 { bottom: 0; left: 0; border-right: none; border-top: none; }
  .facebox.stranger .c2 { bottom: 0; right: 0; border-left: none; border-top: none; }
  .facebox.stranger { animation: scanline 2.2s ease-in-out infinite; }
  .facebox .label {
    position: absolute; left: 50%; transform: translateX(-50%);
    top: calc(100% + 7px); white-space: nowrap;
    font-family: var(--display); font-size: 12px; letter-spacing: 0.08em;
  }
  .facebox.stranger .label { color: var(--cyan); }
  .facebox.known .label { color: var(--amber-soft); opacity: 0.85; font-weight: 300; }
  @keyframes scanline { 50% { opacity: 0.55; } }

  /* ---- companion rail ---- */
  .rail {
    width: 370px; border-left: 1px solid var(--line); background: var(--panel);
    display: flex; flex-direction: column; min-height: 0;
  }
  .presence {
    display: flex; align-items: center; gap: 16px; padding: 20px;
    border-bottom: 1px solid var(--line);
  }
  .orb {
    width: 58px; height: 58px; border-radius: 50%; flex: none;
    background: radial-gradient(circle at 35% 30%, #ffd9a0, #f2a33c 48%, #5c3f19 85%);
    box-shadow: 0 0 26px rgba(242, 163, 60, 0.35);
    animation: breathe 4.2s ease-in-out infinite;
  }
  .orb.listening {
    box-shadow: 0 0 0 3px var(--bg), 0 0 0 5px var(--cyan), 0 0 26px rgba(83,214,224,.35);
  }
  .orb.speaking { animation-duration: 1.3s; box-shadow: 0 0 44px rgba(242,163,60,.6); }
  .orb.thinking { animation-duration: 2.2s; filter: saturate(0.75) brightness(0.9); }
  @keyframes breathe { 50% { transform: scale(1.07); } }
  .presence .who { font-family: var(--display); font-size: 15px; letter-spacing: 0.2em;
    text-transform: uppercase; }
  .presence .how { color: var(--dim); font-size: 12.5px; margin-top: 3px; }
  .people { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 20px;
    border-bottom: 1px solid var(--line); min-height: 20px; }
  .chip {
    font-family: var(--display); font-size: 11.5px; letter-spacing: 0.06em;
    padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line);
    color: var(--dim);
  }
  .chip.known { color: var(--amber-soft); border-color: rgba(242,163,60,.4); }
  .chip.scan { color: var(--cyan); border-color: rgba(83,214,224,.4); }
  .transcript { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .line { margin-bottom: 12px; max-width: 92%; }
  .line .meta { font-family: var(--mono); font-size: 10.5px; color: var(--dim);
    letter-spacing: 0.08em; }
  .line .txt { margin-top: 2px; font-size: 14px; line-height: 1.45; }
  .line.aria { margin-left: auto; text-align: right; }
  .line.aria .txt { color: var(--amber-soft); }
  .line.user .txt { color: var(--cream); }
  .empty { color: var(--dim); font-size: 13px; padding: 8px 2px; }
  footer { padding: 10px 20px; border-top: 1px solid var(--line);
    color: var(--dim); font-size: 11.5px; font-family: var(--mono); }

  @media (prefers-reduced-motion: reduce) {
    .orb, .facebox.stranger { animation: none; }
  }
</style>
</head>
<body>
<header>
  <h1>ARIA</h1>
  <span class="sub">companion &middot; local &middot; webcam</span>
  <span class="status" id="fps">connecting&hellip;</span>
</header>
<main>
  <div class="stage-wrap">
    <div class="stage" id="stage">
      <img src="/stream" alt="ARIA camera view">
      <div id="boxes"></div>
    </div>
  </div>
  <aside class="rail">
    <div class="presence">
      <div class="orb" id="orb"></div>
      <div>
        <div class="who">ARIA</div>
        <div class="how" id="state">idle</div>
      </div>
    </div>
    <div class="people" id="people"></div>
    <div class="transcript" id="transcript" aria-live="polite">
      <div class="empty">Nothing said yet — talk to ARIA.</div>
    </div>
    <footer>127.0.0.1 &middot; nothing leaves this machine but ARIA's own voice calls</footer>
  </aside>
</main>
<script>
const STATE_TEXT = {
  idle: "idle — watching the room",
  listening: "listening…",
  thinking: "thinking…",
  speaking: "speaking",
};
let lastLine = 0;
const transcriptEl = document.getElementById("transcript");

function renderBoxes(faces) {
  const host = document.getElementById("boxes");
  host.innerHTML = "";
  for (const f of faces) {
    const d = document.createElement("div");
    d.className = "facebox " + (f.kind === "known" ? "known" : "stranger");
    d.style.left = (f.x * 100) + "%";
    d.style.top = (f.y * 100) + "%";
    d.style.width = (f.w * 100) + "%";
    d.style.height = (f.h * 100) + "%";
    const c1 = document.createElement("span"); c1.className = "c1";
    const c2 = document.createElement("span"); c2.className = "c2";
    d.appendChild(c1); d.appendChild(c2);
    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = f.kind === "known"
      ? `${f.name} · ${Math.round((f.score || 0) * 100)}%`
      : "scanning";
    d.appendChild(lab);
    host.appendChild(d);
  }
}

function renderPeople(people) {
  const host = document.getElementById("people");
  host.innerHTML = "";
  if (!people.length) {
    host.innerHTML = '<span class="chip">nobody in view</span>';
    return;
  }
  for (const p of people) {
    const c = document.createElement("span");
    c.className = "chip " + (p.kind === "known" ? "known" : "scan");
    c.textContent = p.kind === "known" ? p.name : `scanning ×${p.count}`;
    host.appendChild(c);
  }
}

function renderTranscript(lines) {
  if (!lines.length) return;
  if (lines[lines.length - 1].i <= lastLine) return;
  const empty = transcriptEl.querySelector(".empty");
  if (empty) empty.remove();
  for (const l of lines) {
    if (l.i <= lastLine) continue;
    const d = document.createElement("div");
    d.className = "line " + (l.role === "user" ? "user" : "aria");
    d.innerHTML = `<div class="meta">${l.role === "user" ? "you" : "aria"} · ${l.t}</div>` +
                  `<div class="txt"></div>`;
    d.querySelector(".txt").textContent = l.text;
    transcriptEl.appendChild(d);
  }
  lastLine = lines[lines.length - 1].i;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

async function poll() {
  try {
    const r = await fetch("/state");
    const s = await r.json();
    document.getElementById("fps").textContent =
      (s.status.fps ? s.status.fps.toFixed(1) + " fps" : "—") +
      (s.status.yolo ? " · yolo " + s.status.yolo : "");
    const orb = document.getElementById("orb");
    orb.className = "orb " + s.aria_state;
    document.getElementById("state").textContent = STATE_TEXT[s.aria_state] || s.aria_state;
    renderBoxes(s.faces);
    renderPeople(s.people);
    renderTranscript(s.transcript);
  } catch (e) { /* server restarting — poll again */ }
}
poll();
setInterval(poll, 500);
</script>
</body>
</html>
"""


def serve(hub: UiHub, port: int | None = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the UI server bound to 127.0.0.1 on a daemon thread."""
    port = port or int(os.environ.get("ARIA_UI_PORT", "8080"))

    class Bound(_Handler):
        pass

    Bound.hub = hub
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Bound)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="AriaWebUI")
    t.start()
    return httpd, t
