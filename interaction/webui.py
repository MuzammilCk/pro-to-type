"""ARIA web UI — local dashboard replacing the cv2.imshow debug window.

Stdlib-only (http.server + cv2 JPEG encoding): no new dependencies.

Architecture:
  - Vision thread (run.py) pushes JPEG frames + state snapshots into UiHub
  - GET /stream  — MJPEG video feed (pure vision frame, no cv2 scribbles)
  - GET /state   — JSON: faces (normalized boxes), people, transcript,
                   ARIA's conversational state (idle/listening/thinking/speaking)
  - GET /        — the UI: video stage with DOM overlays + companion rail

Design contract:
  - Obsidian & Dark Amber palette with Cyber-Cyan telemetry & Emerald verification
  - Multi-layer luminous AI Companion Voice Orb with state-reactive plasma rotation
  - Procedural 28-bar acoustic waveform equalizer responding to speech turns
  - High-precision optical reticles with smooth interpolation and laser scanlines
  - Agent Chat stream with distinct ARIA / User glassmorphic message cards
"""
import json
import os
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
<title>ARIA // Autonomous Neural Perception Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #08080a;
    --bg-surface: rgba(18, 17, 24, 0.75);
    --bg-surface-elevated: rgba(28, 26, 36, 0.85);
    --bg-panel: #0d0c12;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-medium: rgba(255, 255, 255, 0.15);
    --border-highlight: rgba(245, 158, 11, 0.4);
    
    --amber: #f59e0b;
    --amber-light: #fbbf24;
    --amber-soft: #fde68a;
    --amber-glow: rgba(245, 158, 11, 0.35);
    
    --cyan: #06b6d4;
    --cyan-light: #22d3ee;
    --cyan-glow: rgba(6, 182, 212, 0.35);
    
    --emerald: #10b981;
    --emerald-light: #34d399;
    --emerald-glow: rgba(16, 185, 129, 0.35);
    
    --rose: #f43f5e;
    --violet: #8b5cf6;
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --font-display: "Outfit", system-ui, sans-serif;
    --font-mono: "JetBrains Mono", Cascadia Code, monospace;
    
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --radius-pill: 9999px;
  }

  @property --angle {
    syntax: "<angle>";
    inherits: false;
    initial-value: 0deg;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; width: 100%; overflow: hidden; }

  body {
    background: var(--bg-base);
    color: var(--text-main);
    font-family: var(--font-display);
    display: flex;
    flex-direction: column;
    position: relative;
    user-select: none;
  }

  /* Ambient Lighting Backdrops */
  .ambient-glow-amber {
    position: fixed;
    top: -150px;
    right: -100px;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.08) 0%, transparent 70%);
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
  }
  .ambient-glow-cyan {
    position: fixed;
    bottom: -200px;
    left: 10%;
    width: 700px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.05) 0%, transparent 70%);
    filter: blur(100px);
    pointer-events: none;
    z-index: 0;
  }
  .bg-grid {
    position: fixed;
    inset: 0;
    background-image: 
      radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
  }

  .app-shell {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  /* ---- Header Bar ---- */
  header.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: rgba(13, 12, 18, 0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-subtle);
    gap: 16px;
    flex: none;
  }

  .brand-block {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-logo-hex {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(6, 182, 212, 0.25));
    border: 1px solid var(--border-highlight);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 16px var(--amber-glow);
    position: relative;
    overflow: hidden;
  }
  .brand-logo-hex::before {
    content: "◈";
    font-size: 20px;
    color: var(--amber-light);
    animation: pulseLogo 3s ease-in-out infinite;
  }
  @keyframes pulseLogo {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 4px var(--amber)); }
    50% { transform: scale(1.15); filter: drop-shadow(0 0 10px var(--amber-light)); }
  }

  .brand-text h1 {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.1;
  }
  .brand-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: var(--radius-pill);
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: var(--amber-soft);
  }
  .brand-text .subtext {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    letter-spacing: 0.08em;
  }

  /* Telemetry Pill Strip */
  .telemetry-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .tele-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    padding: 5px 12px;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: 11.5px;
    transition: border-color 0.2s;
  }
  .tele-pill:hover {
    border-color: var(--border-medium);
  }
  .tele-label {
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 500;
  }
  .tele-val {
    color: var(--text-main);
    font-weight: 600;
  }
  .status-indicator {
    width: 7px;
    height: 7px;
    border-radius: 50%;
  }
  .status-indicator.green {
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald-glow);
    animation: statusBlink 2.5s infinite;
  }
  .status-indicator.cyan {
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan-glow);
  }
  .status-indicator.amber {
    background: var(--amber);
    box-shadow: 0 0 8px var(--amber-glow);
  }
  @keyframes statusBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Header Controls */
  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .btn-icon-control {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .btn-icon-control:hover {
    background: rgba(255, 255, 255, 0.09);
    border-color: var(--border-medium);
    color: #fff;
    transform: translateY(-1px);
  }
  .btn-icon-control.active {
    background: rgba(245, 158, 11, 0.15);
    border-color: var(--amber);
    color: var(--amber-light);
  }

  /* ---- Main Dashboard Layout ---- */
  main.dashboard-body {
    flex: 1;
    display: flex;
    min-height: 0;
    overflow: hidden;
  }

  /* ---- Stage Section (Left 68%) ---- */
  .vision-deck {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 18px 20px;
    min-width: 0;
    overflow: hidden;
    gap: 12px;
  }

  .deck-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4px;
    flex: none;
  }
  .deck-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .deck-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-main);
  }
  .live-rec-tag {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    color: var(--rose);
    letter-spacing: 0.1em;
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.3);
    padding: 2px 8px;
    border-radius: var(--radius-pill);
  }
  .live-rec-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--rose);
    box-shadow: 0 0 6px var(--rose);
    animation: recPulse 1.2s ease-in-out infinite;
  }
  @keyframes recPulse { 50% { opacity: 0.25; } }

  .deck-meta {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    gap: 12px;
  }

  /* Stage Viewport */
  .viewport-container {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(9, 8, 14, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }

  /* HUD Framing Brackets */
  .hud-bracket {
    position: absolute;
    width: 22px;
    height: 22px;
    pointer-events: none;
    z-index: 10;
  }
  .hud-bracket.tl { top: 14px; left: 14px; border-top: 2px solid var(--amber); border-left: 2px solid var(--amber); }
  .hud-bracket.tr { top: 14px; right: 14px; border-top: 2px solid var(--amber); border-right: 2px solid var(--amber); }
  .hud-bracket.bl { bottom: 14px; left: 14px; border-bottom: 2px solid var(--cyan); border-left: 2px solid var(--cyan); }
  .hud-bracket.br { bottom: 14px; right: 14px; border-bottom: 2px solid var(--cyan); border-right: 2px solid var(--cyan); }

  .video-stage-wrap {
    position: relative;
    display: inline-block;
    max-width: 100%;
    max-height: 100%;
  }

  .video-stage-wrap img.feed-image {
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 210px);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    object-fit: contain;
  }

  /* Reticle Overlay */
  .boxes-layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .facebox {
    position: absolute;
    pointer-events: none;
    transition: left 0.16s ease-out, top 0.16s ease-out, width 0.16s ease-out, height 0.16s ease-out;
  }

  /* Known Face Reticle */
  .facebox.known {
    border: 1.5px solid rgba(16, 185, 129, 0.85);
    border-radius: var(--radius-sm);
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.35), inset 0 0 12px rgba(16, 185, 129, 0.15);
  }
  .facebox.known .c-marker {
    position: absolute;
    width: 6px;
    height: 6px;
    background: var(--emerald);
  }
  .facebox.known .c-marker.tl { top: -2px; left: -2px; }
  .facebox.known .c-marker.tr { top: -2px; right: -2px; }
  .facebox.known .c-marker.bl { bottom: -2px; left: -2px; }
  .facebox.known .c-marker.br { bottom: -2px; right: -2px; }

  .facebox.known .id-card {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(12, 22, 18, 0.9);
    border: 1px solid rgba(16, 185, 129, 0.6);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6);
    border-radius: 6px;
    padding: 3px 10px;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
  }
  .id-card .id-name {
    font-size: 11.5px;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .id-card .id-score {
    font-size: 10px;
    color: var(--emerald-light);
    font-weight: 600;
  }
  .id-card .id-badge {
    font-size: 9px;
    padding: 1px 4px;
    background: rgba(16, 185, 129, 0.25);
    border-radius: 3px;
    color: #a7f3d0;
  }

  /* Stranger Face Reticle */
  .facebox.stranger {
    border: 1px dashed rgba(6, 182, 212, 0.4);
    animation: strangerPulse 2s ease-in-out infinite;
  }
  @keyframes strangerPulse {
    0%, 100% { opacity: 0.9; }
    50% { opacity: 0.6; }
  }
  .facebox.stranger .bracket {
    position: absolute;
    width: 14px;
    height: 14px;
    border: 2px solid var(--cyan-light);
  }
  .facebox.stranger .bracket.tl { top: 0; left: 0; border-right: none; border-bottom: none; }
  .facebox.stranger .bracket.tr { top: 0; right: 0; border-left: none; border-bottom: none; }
  .facebox.stranger .bracket.bl { bottom: 0; left: 0; border-right: none; border-top: none; }
  .facebox.stranger .bracket.br { bottom: 0; right: 0; border-left: none; border-top: none; }

  .facebox.stranger .laser-scan {
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent, rgba(6, 182, 212, 0.25), transparent);
    animation: laserMotion 2.4s ease-in-out infinite;
  }
  @keyframes laserMotion {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
  }

  .facebox.stranger .id-card {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(6, 20, 26, 0.9);
    border: 1px solid rgba(6, 182, 212, 0.6);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6);
    border-radius: 6px;
    padding: 3px 10px;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
  }
  .facebox.stranger .id-card .id-name {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--cyan-light);
    letter-spacing: 0.08em;
  }
  .facebox.stranger .id-card .radar-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    animation: recPulse 0.9s infinite;
  }

  /* Environmental Awareness Strip */
  .object-telemetry-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(14, 13, 19, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    padding: 10px 18px;
    border-radius: var(--radius-md);
    flex: none;
  }
  .tele-title {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .object-pills-list {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    overflow-x: auto;
  }
  .obj-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    font-size: 11.5px;
    color: var(--text-main);
    transition: all 0.2s;
  }
  .obj-pill:hover {
    border-color: var(--border-medium);
    background: rgba(255, 255, 255, 0.07);
  }
  .obj-icon { font-size: 12px; }
  .obj-empty {
    color: var(--text-dim);
    font-size: 11.5px;
    font-style: italic;
  }

  /* ---- Companion Rail (Right 32%) ---- */
  aside.companion-console {
    width: 390px;
    background: rgba(13, 12, 18, 0.9);
    backdrop-filter: blur(24px);
    border-left: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: none;
    overflow: hidden;
  }

  /* Presence & Voice Deck */
  .voice-presence-deck {
    padding: 22px 20px 18px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.04) 0%, transparent 100%);
    position: relative;
  }

  /* Multi-layer Conic AI Orb */
  .orb-stage-wrapper {
    position: relative;
    width: 110px;
    height: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .orb-ambient-aura {
    position: absolute;
    inset: -12px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--amber-glow) 0%, transparent 70%);
    filter: blur(20px);
    transition: all 0.5s ease;
    pointer-events: none;
  }

  .orb-orbital-ring {
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    border: 1px dashed rgba(245, 158, 11, 0.35);
    animation: orbitRotate 20s linear infinite;
    pointer-events: none;
  }
  @keyframes orbitRotate {
    to { transform: rotate(360deg); }
  }

  .companion-orb {
    width: 82px;
    height: 82px;
    border-radius: 50%;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 32px var(--amber-glow), inset 0 0 16px rgba(255, 255, 255, 0.3);
    animation: orbBreathe 4s ease-in-out infinite;
    cursor: pointer;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  .companion-orb:hover {
    transform: scale(1.08);
  }

  .companion-orb::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(
      from var(--angle) at 50% 50%,
      #ffd9a0 0deg,
      #f59e0b 90deg,
      #d97706 180deg,
      #f59e0b 270deg,
      #ffd9a0 360deg
    );
    filter: blur(6px) contrast(1.3);
    animation: rotateOrb 8s linear infinite;
  }

  .companion-orb::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255, 255, 255, 0.6) 0%, transparent 60%);
    mix-blend-mode: overlay;
  }

  @keyframes rotateOrb {
    to { --angle: 360deg; }
  }
  @keyframes orbBreathe {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }

  /* Orb State Variants */
  .companion-orb.listening {
    box-shadow: 0 0 36px var(--cyan-glow), inset 0 0 20px rgba(34, 211, 238, 0.6);
  }
  .companion-orb.listening::before {
    background: conic-gradient(
      from var(--angle) at 50% 50%,
      #a5f3fc 0deg,
      #06b6d4 90deg,
      #0284c7 180deg,
      #22d3ee 270deg,
      #a5f3fc 360deg
    );
    animation-duration: 3s;
  }
  .orb-ambient-aura.listening {
    background: radial-gradient(circle, var(--cyan-glow) 0%, transparent 70%);
  }

  .companion-orb.thinking {
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.5), inset 0 0 20px rgba(236, 72, 153, 0.5);
    animation: orbBreathe 1.8s ease-in-out infinite;
  }
  .companion-orb.thinking::before {
    background: conic-gradient(
      from var(--angle) at 50% 50%,
      #c4b5fd 0deg,
      #8b5cf6 90deg,
      #f59e0b 180deg,
      #ec4899 270deg,
      #c4b5fd 360deg
    );
    animation-duration: 2.2s;
  }
  .orb-ambient-aura.thinking {
    background: radial-gradient(circle, rgba(139, 92, 246, 0.4) 0%, transparent 70%);
  }

  .companion-orb.speaking {
    box-shadow: 0 0 48px var(--amber-glow), inset 0 0 24px rgba(255, 237, 213, 0.8);
    animation: orbSpeak 1.2s ease-in-out infinite;
  }
  .companion-orb.speaking::before {
    background: conic-gradient(
      from var(--angle) at 50% 50%,
      #fffbeb 0deg,
      #f97316 90deg,
      #fbbf24 180deg,
      #ea580c 270deg,
      #fffbeb 360deg
    );
    animation-duration: 1.4s;
  }
  @keyframes orbSpeak {
    0%, 100% { transform: scale(1); filter: brightness(1); }
    50% { transform: scale(1.1); filter: brightness(1.25); }
  }

  /* Presence Title & Subtitle */
  .voice-state-meta {
    text-align: center;
  }
  .state-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--amber-light);
    text-transform: uppercase;
  }
  .state-pill-badge.listening {
    background: rgba(6, 182, 212, 0.15);
    border-color: rgba(6, 182, 212, 0.4);
    color: var(--cyan-light);
  }
  .state-pill-badge.thinking {
    background: rgba(139, 92, 246, 0.15);
    border-color: rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
  }
  .state-pill-badge.speaking {
    background: rgba(245, 158, 11, 0.25);
    border-color: var(--amber);
    color: #fff;
    box-shadow: 0 0 10px var(--amber-glow);
  }

  .state-subtext {
    font-size: 11.5px;
    color: var(--text-dim);
    margin-top: 5px;
    font-family: var(--font-mono);
  }

  /* Procedural Acoustic Waveform */
  .acoustic-waveform-panel {
    width: 100%;
    background: rgba(18, 17, 24, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 8px 12px 10px;
    position: relative;
    overflow: hidden;
  }
  .waveform-top-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .waveform-top-meta .wave-live-tag {
    color: var(--emerald-light);
    font-weight: 700;
  }
  .waveform-bars-container {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    height: 32px;
    gap: 2px;
    padding: 0 2px;
  }
  .wave-bar {
    flex: 1;
    min-width: 2px;
    background: var(--amber);
    border-radius: 2px 2px 0 0;
    transition: height 0.08s ease, background 0.2s;
    opacity: 0.75;
  }
  .wave-bar.speaking {
    background: linear-gradient(180deg, #fde68a, #f59e0b);
    box-shadow: 0 0 6px var(--amber-glow);
    opacity: 1;
  }
  .wave-bar.listening {
    background: linear-gradient(180deg, #a5f3fc, #06b6d4);
    box-shadow: 0 0 6px var(--cyan-glow);
    opacity: 1;
  }

  /* Presence Attendee Deck */
  .attendees-card {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: none;
  }
  .attendees-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 10.5px;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .attendees-chips-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .person-chip {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    padding: 3px 10px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--border-subtle);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
  }
  .person-chip.known {
    color: var(--emerald-light);
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.1);
  }
  .person-chip.scan {
    color: var(--cyan-light);
    border-color: rgba(6, 182, 212, 0.4);
    background: rgba(6, 182, 212, 0.1);
  }

  /* Conversational Stream (Transcript) */
  .chat-stream-deck {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }

  .chat-deck-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px 10px;
    border-bottom: 1px solid var(--border-subtle);
    flex: none;
  }
  .chat-title-group {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-main);
  }
  .chat-count-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 1px 6px;
    border-radius: var(--radius-pill);
    background: rgba(255, 255, 255, 0.07);
    color: var(--text-dim);
  }
  .chat-deck-tools {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .mini-tool-btn {
    background: transparent;
    border: 1px solid var(--border-subtle);
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s;
  }
  .mini-tool-btn:hover {
    color: var(--text-main);
    border-color: var(--border-medium);
  }

  .chat-history-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scroll-behavior: smooth;
  }
  .chat-history-scroll::-webkit-scrollbar {
    width: 5px;
  }
  .chat-history-scroll::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
  }

  /* Chat Messages */
  .chat-bubble {
    display: flex;
    flex-direction: column;
    max-width: 90%;
    animation: bubbleEnter 0.25s ease-out;
  }
  @keyframes bubbleEnter {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .chat-bubble.user {
    align-self: flex-start;
  }
  .chat-bubble.user .msg-content {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    color: var(--text-main);
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px;
  }

  .chat-bubble.aria {
    align-self: flex-end;
  }
  .chat-bubble.aria .msg-content {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.35);
    box-shadow: 0 4px 18px rgba(245, 158, 11, 0.06);
    color: #fff;
    border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
  }

  .msg-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 3px;
    text-transform: uppercase;
  }
  .chat-bubble.aria .msg-meta { justify-content: flex-end; color: var(--amber-soft); }

  .msg-content {
    padding: 9px 13px;
    font-size: 13.5px;
    line-height: 1.45;
    word-break: break-word;
    letter-spacing: 0.01em;
  }

  .chat-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%;
    color: var(--text-dim);
    padding: 30px 20px;
    gap: 10px;
  }
  .chat-empty-icon {
    font-size: 28px;
    opacity: 0.5;
  }
  .chat-empty-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
  }
  .chat-empty-desc {
    font-size: 11.5px;
    line-height: 1.5;
    max-width: 240px;
  }

  /* Thinking Synapse Wave */
  .thinking-indicator-row {
    padding: 6px 20px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--amber-soft);
    flex: none;
  }
  .synapse-dots {
    display: flex;
    gap: 4px;
  }
  .synapse-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--amber);
    animation: synapsePulse 1.2s infinite;
  }
  .synapse-dot:nth-child(2) { animation-delay: 0.2s; }
  .synapse-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes synapsePulse {
    0%, 100% { transform: scale(0.6); opacity: 0.3; }
    50% { transform: scale(1.2); opacity: 1; }
  }

  /* Rail Footer */
  footer.rail-footer-bar {
    padding: 10px 20px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    background: rgba(9, 8, 14, 0.7);
    flex: none;
  }
  .privacy-tag {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  kbd {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--border-subtle);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9px;
  }

  @media (prefers-reduced-motion: reduce) {
    .companion-orb, .companion-orb::before, .orb-orbital-ring { animation: none !important; }
    .facebox.stranger .laser-scan { animation: none !important; }
  }

  @media (max-width: 900px) {
    main.dashboard-body { flex-direction: column; overflow-y: auto; }
    aside.companion-console { width: 100%; height: 500px; }
  }
</style>
</head>
<body>
<div class="ambient-glow-amber"></div>
<div class="ambient-glow-cyan"></div>
<div class="bg-grid"></div>

<div class="app-shell">
  <!-- Top Bar -->
  <header class="navbar">
    <div class="brand-block">
      <div class="brand-logo-hex"></div>
      <div class="brand-text">
        <h1>ARIA <span class="brand-badge">2.5 CORE</span></h1>
        <div class="subtext">LOCAL COMPANION // VISION & VOICE INTELLIGENCE</div>
      </div>
    </div>

    <div class="telemetry-strip">
      <div class="tele-pill">
        <span class="status-indicator green"></span>
        <span class="tele-label">SYSTEM</span>
        <span class="tele-val" id="tele-engine">ONLINE</span>
      </div>
      <div class="tele-pill">
        <span class="status-indicator cyan"></span>
        <span class="tele-label">VISION</span>
        <span class="tele-val" id="tele-fps">-- FPS</span>
      </div>
      <div class="tele-pill">
        <span class="tele-label">YOLO ENGINE</span>
        <span class="tele-val" id="tele-yolo">2.0s</span>
      </div>
      <div class="tele-pill">
        <span class="status-indicator amber"></span>
        <span class="tele-label">SESSION</span>
        <span class="tele-val">127.0.0.1:8200</span>
      </div>
    </div>

    <div class="header-actions">
      <button class="btn-icon-control" id="btn-sound" title="Toggle UI Audio SFX (M)">
        <span id="sound-icon">🔇</span>
        <span>SOUND</span>
      </button>
      <button class="btn-icon-control" id="btn-hud" title="Toggle HUD Overlays (H)">
        <span>🎯</span>
        <span>HUD</span>
      </button>
      <button class="btn-icon-control" id="btn-fs" title="Toggle Fullscreen (F)">
        <span>⛶</span>
        <span>EXPAND</span>
      </button>
    </div>
  </header>

  <!-- Main Body -->
  <main class="dashboard-body">
    <!-- Left: Vision Deck -->
    <section class="vision-deck" id="vision-deck">
      <div class="deck-header">
        <div class="deck-title-group">
          <span class="deck-title">NEURAL VISION VIEWPORT</span>
          <span class="live-rec-tag"><span class="live-rec-dot"></span>SENSOR STREAM</span>
        </div>
        <div class="deck-meta">
          <span>OPTIC: CAM_0</span>
          <span>RESOLUTION: 640×480</span>
          <span id="entity-stat-label">ENTITIES: 0</span>
        </div>
      </div>

      <!-- Viewport Stage -->
      <div class="viewport-container" id="viewport-box">
        <div class="hud-bracket tl"></div>
        <div class="hud-bracket tr"></div>
        <div class="hud-bracket bl"></div>
        <div class="hud-bracket br"></div>

        <div class="video-stage-wrap" id="stage">
          <img src="/stream" alt="ARIA camera stream" class="feed-image" id="feed-img">
          <div id="boxes" class="boxes-layer"></div>
        </div>
      </div>

      <!-- Object Telemetry Strip -->
      <div class="object-telemetry-bar">
        <span class="tele-title">SCENE RECOGNITION:</span>
        <div class="object-pills-list" id="objects-list">
          <span class="obj-empty">Observing room for physical objects...</span>
        </div>
      </div>
    </section>

    <!-- Right: Companion Rail -->
    <aside class="companion-console">
      <!-- Presence & Orb -->
      <div class="voice-presence-deck">
        <div class="orb-stage-wrapper">
          <div class="orb-ambient-aura" id="orb-aura"></div>
          <div class="orb-orbital-ring"></div>
          <div class="companion-orb" id="orb" title="ARIA Voice Presence Core"></div>
        </div>

        <div class="voice-state-meta">
          <div class="state-pill-badge" id="state-pill">IDLE · OBSERVING</div>
          <div class="state-subtext" id="state-sub">Observing room & listening</div>
        </div>

        <!-- Waveform Visualizer -->
        <div class="acoustic-waveform-panel">
          <div class="waveform-top-meta">
            <span>ACOUSTIC HARMONICS</span>
            <span class="wave-live-tag" id="wave-state-tag">MONITORING</span>
          </div>
          <div class="waveform-bars-container" id="wave-bars"></div>
        </div>
      </div>

      <!-- People Radar -->
      <div class="attendees-card">
        <div class="attendees-header">
          <span>PEOPLE IN VIEW</span>
          <span id="people-count">0 ACTIVE</span>
        </div>
        <div class="attendees-chips-wrap" id="people">
          <span class="person-chip">Nobody in view</span>
        </div>
      </div>

      <!-- Chat Stream -->
      <div class="chat-stream-deck">
        <div class="chat-deck-header">
          <div class="chat-title-group">
            <span>CONVERSATION STREAM</span>
            <span class="chat-count-tag" id="chat-count">0</span>
          </div>
          <div class="chat-deck-tools">
            <button class="mini-tool-btn" id="btn-clear-chat" title="Clear Dialogue View">CLEAR</button>
          </div>
        </div>

        <div class="chat-history-scroll" id="transcript" aria-live="polite">
          <div class="chat-empty-state" id="chat-empty">
            <div class="chat-empty-icon">🎙</div>
            <div class="chat-empty-title">Awaiting Spoken Turn</div>
            <div class="chat-empty-desc">Say hello to ARIA. Every spoken exchange is transcribed here in real-time.</div>
          </div>
        </div>

        <div class="thinking-indicator-row" id="thinking-indicator" style="display: none;">
          <div class="synapse-dots">
            <div class="synapse-dot"></div>
            <div class="synapse-dot"></div>
            <div class="synapse-dot"></div>
          </div>
          <span>ARIA is formulating response...</span>
        </div>
      </div>

      <!-- Footer -->
      <footer class="rail-footer-bar">
        <div class="privacy-tag">
          <span>🔒</span>
          <span>127.0.0.1 LOCAL SECURE CORE</span>
        </div>
        <div>
          <kbd>F</kbd> Expand &nbsp; <kbd>H</kbd> HUD &nbsp; <kbd>M</kbd> Sound
        </div>
      </footer>
    </aside>
  </main>
</div>

<script>
// --- State & Config ---
const STATE_META = {
  idle: { label: "IDLE · OBSERVING", sub: "Observing room & ready for speech", wave: "PASSIVE", class: "" },
  listening: { label: "LISTENING...", sub: "Capturing user vocal stream", wave: "ACTIVE IN", class: "listening" },
  thinking: { label: "THINKING...", sub: "Synthesizing reasoning & context", wave: "PROCESSING", class: "thinking" },
  speaking: { label: "SPEAKING", sub: "Vocalizing companion response", wave: "VOCAL HARMONICS", class: "speaking" },
};

const OBJECT_ICONS = {
  person: "👤", chair: "🪑", couch: "🛋", sofa: "🛋", tv: "🖥", monitor: "🖥",
  laptop: "💻", cell_phone: "📱", "cell phone": "📱", phone: "📱", bottle: "🍾",
  cup: "☕", book: "📖", mouse: "🖱", keyboard: "⌨", backpack: "🎒", handbag: "👜",
  suitcase: "🧳", clock: "⏰", default: "📦"
};

let lastLine = 0;
let totalMessages = 0;
let soundEnabled = false;
let hudEnabled = true;
let currentState = "idle";

// --- Audio Synthesizer (Web Audio API) ---
let audioCtx = null;
function playUiTone(freq = 440, type = "sine", duration = 0.12, gainVal = 0.08) {
  if (!soundEnabled) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

// --- Procedural Waveform Visualizer (28 Bars) ---
const waveContainer = document.getElementById("wave-bars");
const NUM_BARS = 28;
const bars = [];
for (let i = 0; i < NUM_BARS; i++) {
  const b = document.createElement("div");
  b.className = "wave-bar";
  b.style.height = "4px";
  waveContainer.appendChild(b);
  bars.push(b);
}

let waveFrame = 0;
function animateWaveform() {
  waveFrame++;
  const f = waveFrame;
  let minH = 3, maxH = 8;
  if (currentState === "listening") { minH = 6; maxH = 26; }
  else if (currentState === "thinking") { minH = 5; maxH = 18; }
  else if (currentState === "speaking") { minH = 8; maxH = 32; }

  for (let i = 0; i < NUM_BARS; i++) {
    let combined = 0;
    if (currentState === "speaking") {
      const w1 = Math.sin(f * 0.08 + i * 0.45) * 0.5 + 0.5;
      const w2 = Math.sin(f * 0.14 + i * 0.25 + 1.2) * 0.35 + 0.35;
      const noise = Math.random() * 0.3;
      combined = (w1 + w2 + noise) / 1.7;
    } else if (currentState === "listening") {
      const w1 = Math.sin(f * 0.06 + i * 0.3) * 0.5 + 0.5;
      const noise = Math.random() * 0.4;
      combined = (w1 + noise) / 1.4;
    } else if (currentState === "thinking") {
      combined = Math.sin(f * 0.07 + i * 0.2) * 0.5 + 0.5;
    } else {
      combined = Math.sin(f * 0.03 + i * 0.15) * 0.5 + 0.5;
    }
    const h = Math.round(minH + (maxH - minH) * combined);
    bars[i].style.height = h + "px";
    bars[i].className = "wave-bar " + (currentState === "speaking" ? "speaking" : (currentState === "listening" ? "listening" : ""));
  }
  requestAnimationFrame(animateWaveform);
}
requestAnimationFrame(animateWaveform);

// --- Face Box Reticles ---
function renderBoxes(faces) {
  const host = document.getElementById("boxes");
  if (!hudEnabled) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = "";
  document.getElementById("entity-stat-label").textContent = `ENTITIES: ${faces.length}`;

  for (const f of faces) {
    const isKnown = f.kind === "known";
    const d = document.createElement("div");
    d.className = "facebox " + (isKnown ? "known" : "stranger");
    d.style.left = (f.x * 100) + "%";
    d.style.top = (f.y * 100) + "%";
    d.style.width = (f.w * 100) + "%";
    d.style.height = (f.h * 100) + "%";

    if (isKnown) {
      ["tl", "tr", "bl", "br"].forEach(pos => {
        const m = document.createElement("span");
        m.className = "c-marker " + pos;
        d.appendChild(m);
      });
      const card = document.createElement("div");
      card.className = "id-card";
      const scorePct = Math.round((f.score || 0) * 100);
      card.innerHTML = `<span class="id-badge">✓</span>` +
                       `<span class="id-name">${f.name || "Known"}</span>` +
                       `<span class="id-score">${scorePct}%</span>`;
      d.appendChild(card);
    } else {
      ["tl", "tr", "bl", "br"].forEach(pos => {
        const b = document.createElement("span");
        b.className = "bracket " + pos;
        d.appendChild(b);
      });
      const laser = document.createElement("div");
      laser.className = "laser-scan";
      d.appendChild(laser);

      const card = document.createElement("div");
      card.className = "id-card";
      card.innerHTML = `<span class="radar-dot"></span><span class="id-name">SCANNING VISITOR</span>`;
      d.appendChild(card);
    }
    host.appendChild(d);
  }
}

// --- People In View ---
function renderPeople(people) {
  const host = document.getElementById("people");
  host.innerHTML = "";
  if (!people || !people.length) {
    host.innerHTML = '<span class="person-chip">Nobody in view</span>';
    document.getElementById("people-count").textContent = "0 ACTIVE";
    return;
  }
  let count = 0;
  for (const p of people) {
    const isKnown = p.kind === "known";
    const c = document.createElement("span");
    c.className = "person-chip " + (isKnown ? "known" : "scan");
    if (isKnown) {
      c.innerHTML = `<span>●</span><span>${p.name}</span>`;
      count++;
    } else {
      c.innerHTML = `<span>◌</span><span>Stranger ×${p.count || 1}</span>`;
      count += (p.count || 1);
    }
    host.appendChild(c);
  }
  document.getElementById("people-count").textContent = `${count} ACTIVE`;
}

// --- Scene Objects ---
function renderObjects(objects) {
  const host = document.getElementById("objects-list");
  host.innerHTML = "";
  if (!objects || !objects.length) {
    host.innerHTML = '<span class="obj-empty">Observing room for physical objects...</span>';
    return;
  }
  for (const obj of objects) {
    const pill = document.createElement("span");
    pill.className = "obj-pill";
    const icon = OBJECT_ICONS[obj.toLowerCase()] || OBJECT_ICONS.default;
    pill.innerHTML = `<span class="obj-icon">${icon}</span><span>${obj}</span>`;
    host.appendChild(pill);
  }
}

// --- Chat Transcript ---
const transcriptEl = document.getElementById("transcript");
function renderTranscript(lines) {
  if (!lines || !lines.length) return;
  if (lines[lines.length - 1].i <= lastLine) return;

  const empty = document.getElementById("chat-empty");
  if (empty) empty.remove();

  for (const l of lines) {
    if (l.i <= lastLine) continue;
    totalMessages++;
    const isUser = l.role === "user";
    const d = document.createElement("div");
    d.className = "chat-bubble " + (isUser ? "user" : "aria");
    d.innerHTML = `
      <div class="msg-meta">
        <span>${isUser ? "👤 YOU" : "◈ ARIA"}</span>
        <span>·</span>
        <span>${l.t || ""}</span>
      </div>
      <div class="msg-content"></div>
    `;
    d.querySelector(".msg-content").textContent = l.text;
    transcriptEl.appendChild(d);

    if (!isUser) {
      playUiTone(580, "sine", 0.15, 0.05);
    }
  }
  lastLine = lines[lines.length - 1].i;
  document.getElementById("chat-count").textContent = totalMessages;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// --- Polling & State Engine ---
async function poll() {
  try {
    const r = await fetch("/state");
    const s = await r.json();

    // 1. Telemetry
    const fpsVal = s.status.fps ? s.status.fps.toFixed(1) + " FPS" : "— FPS";
    document.getElementById("tele-fps").textContent = fpsVal;
    if (s.status.yolo) {
      document.getElementById("tele-yolo").textContent = s.status.yolo;
    }

    // 2. ARIA State & Orb
    const state = s.aria_state || "idle";
    if (state !== currentState) {
      const prev = currentState;
      currentState = state;
      const meta = STATE_META[state] || STATE_META.idle;
      const orb = document.getElementById("orb");
      const aura = document.getElementById("orb-aura");
      const statePill = document.getElementById("state-pill");
      
      orb.className = "companion-orb " + meta.class;
      aura.className = "orb-ambient-aura " + meta.class;
      statePill.className = "state-pill-badge " + meta.class;
      statePill.textContent = meta.label;
      document.getElementById("state-sub").textContent = meta.sub;
      document.getElementById("wave-state-tag").textContent = meta.wave;

      const thinkingRow = document.getElementById("thinking-indicator");
      if (state === "thinking") {
        thinkingRow.style.display = "flex";
      } else {
        thinkingRow.style.display = "none";
      }

      if (state === "speaking") {
        playUiTone(620, "sine", 0.18, 0.07);
      } else if (state === "listening" && prev !== "listening") {
        playUiTone(440, "triangle", 0.1, 0.05);
      }
    }

    // 3. Face Reticles, People, Objects, Transcript
    renderBoxes(s.faces || []);
    renderPeople(s.people || []);
    renderObjects(s.status.objects || []);
    renderTranscript(s.transcript || []);
  } catch (e) {
    // Network retry or server reconnect
  }
}

poll();
setInterval(poll, 300);

// --- Controls & Shortcuts ---
document.getElementById("btn-sound").addEventListener("click", () => {
  soundEnabled = !soundEnabled;
  const icon = document.getElementById("sound-icon");
  const btn = document.getElementById("btn-sound");
  icon.textContent = soundEnabled ? "🔊" : "🔇";
  btn.classList.toggle("active", soundEnabled);
  if (soundEnabled) playUiTone(520, "sine", 0.12, 0.08);
});

document.getElementById("btn-hud").addEventListener("click", () => {
  hudEnabled = !hudEnabled;
  const btn = document.getElementById("btn-hud");
  btn.classList.toggle("active", hudEnabled);
  const host = document.getElementById("boxes");
  if (!hudEnabled) host.innerHTML = "";
});

document.getElementById("btn-fs").addEventListener("click", () => {
  const el = document.getElementById("vision-deck");
  if (!document.fullscreenElement) {
    el.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
});

document.getElementById("btn-clear-chat").addEventListener("click", () => {
  transcriptEl.innerHTML = `
    <div class="chat-empty-state" id="chat-empty">
      <div class="chat-empty-icon">🎙</div>
      <div class="chat-empty-title">Transcript Cleared</div>
      <div class="chat-empty-desc">New spoken dialogue will appear here.</div>
    </div>
  `;
  totalMessages = 0;
  document.getElementById("chat-count").textContent = "0";
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "f") {
    document.getElementById("btn-fs").click();
  } else if (k === "h") {
    document.getElementById("btn-hud").click();
  } else if (k === "m") {
    document.getElementById("btn-sound").click();
  } else if (k === "c") {
    document.getElementById("btn-clear-chat").click();
  }
});
</script>
</body>
</html>
"""


def serve(hub: UiHub, port: int | None = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the UI server bound to 127.0.0.1 on a daemon thread."""
    port = port if port is not None else int(os.environ.get("ARIA_UI_PORT", "8200"))

    class Bound(_Handler):
        pass

    Bound.hub = hub
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Bound)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="AriaWebUI")
    t.start()
    return httpd, t
