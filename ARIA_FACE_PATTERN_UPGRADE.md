# ARIA — Face Structure Pattern Upgrade (Recognition HUD)

> **Behavior note (2026-09, confirmed by owner — FINAL).** The "scanning"
> indicator renders on UNRECOGNIZED faces ONLY. It appears the moment a face
> shows up and disappears the moment the system recognizes the person or
> enrolls them. RECOGNIZED faces get NO box and NO mesh — only a small name
> caption. The earlier "green box + name" look was explicitly rejected by
> the owner (it read as clutter). Do NOT reintroduce boxes for authorized
> faces, do NOT "fix" the disappearing indicator as a bug, and note the
> tracker absorbs ghost tracks to prevent the same face ever rendering two
> patterns at once.
>
> **Renderer change (2026-09, web UI).** The indicator is no longer a
> MediaPipe mesh painted by cv2 — it is now animated cyan corner brackets
> drawn by the browser (webui.py, `.facebox.stranger`), positioned from
> normalized track boxes in `/state`. face_pattern.py is retired from the
> live path (kept in repo/history). Same contract, better renderer: real
> typography, zero CPU cost in the vision loop. The vision pipeline no
> longer draws anything onto frames — frames go to the browser untouched.

Read this alongside `ARIA_ARCHITECTURE.md`, but treat this file as self-contained — the repo has moved on since that file was written, so don't assume the exact file layout it describes still holds.

## What "modern face structure pattern" means technically

A rectangle only tells you where a face is. A pattern that follows the actual boundary — jawline, eyes, brows, nose, mouth — is a facial landmark mesh. The library that does this well in real time on CPU is Google's MediaPipe, via its Face Landmarker task (468 3D landmarks per face). This is a different, far richer model than the 5-point landmarks OpenCV's own YuNet detector already returns — YuNet's 5 points (eye corners, nose tip, mouth corners) are enough for face alignment before recognition, but too sparse to draw a convincing structure pattern.

One API trap to avoid: MediaPipe has an old "solutions" API (`mp.solutions.face_mesh`) that still shows up in most tutorials and sample code online, but Google marked it legacy back in 2023 in favor of a new Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`). Reports on whether current package versions still expose the old API are genuinely inconsistent — don't let your agent commit to it just because a tutorial uses it. Build against the Tasks API; it's the one actively maintained and documented going forward.

## Two visual styles — pick one

MediaPipe's drawing utilities ship two built-in styles from the same 468 landmarks:
- **Contours style** — face oval, eyebrows, eyes, lips, nose bridge as clean outlines. Reads as "face boundary," matches what was described, and is cheaper to render (fewer lines).
- **Tesselation style** — the full triangulated mesh across the face surface. Denser, more of a wireframe-scan look, slightly more expensive.

Default to contours for the boundary look described; tesselation is an easy toggle later if a denser look is wanted for the demo video.

## Architecture — how this plugs into what already exists

This is a new component alongside the existing identity pipeline, not a replacement:

- **Identity pipeline (unchanged):** YuNet detects → SFace recognizes → the Presence Manager outputs `recognized` or `unrecognized` per tracked face.
- **New: pattern renderer.** MediaPipe FaceLandmarker runs on the same frame, gets 468 landmarks per face, and draws the pattern. It reads the Presence Manager's existing recognition status for that same tracked face to decide color/visibility:
  - `unrecognized` → draw the pattern every frame
  - `recognized` → stop drawing it

Because "becomes authorized" just means the Presence Manager's status flips from `unrecognized` to `recognized` — which already happens automatically the moment a face matches a stored embedding, including right after someone completes the registration flow — **the pattern hiding itself needs no new logic beyond reading that existing status.** This isn't a new "authorization" concept to build; it's a renderer becoming color-aware of a state that's already computed.

## Performance note for CPU-only hardware

Running YuNet + SFace + FaceLandmarker on every single frame is three models competing for the same CPU. Since the pattern is purely visual (it doesn't feed back into recognition logic), it doesn't need to update every frame to look smooth:
- Run FaceLandmarker every 2nd or 3rd frame, reusing the last known landmark positions in between — imperceptible at normal webcam frame rates, meaningfully cheaper
- Run it only on the region each tracked face already occupies (from the existing tracker), not the full frame — re-scanning the whole image when you already know roughly where the face is wastes cycles

## Step-by-step for your agent

Since the repo has drifted from what was described earlier, Step 1 is not optional — don't let the agent assume the old layout still applies.

**Step 1 — Inspect current state**
Look at the current repo structure, specifically wherever face detection/recognition and frame rendering currently live. Confirm the current Python version in use (`python --version`) and whether a virtual environment already exists.

**Step 2 — Set up the environment**
If not already on Python 3.11/3.12, create a fresh virtual environment on one of those versions for this project. Reinstall existing dependencies (opencv-python, the Sarvam SDK, etc.) into it and confirm the existing detection/recognition pipeline still runs *before* adding anything new — this isolates whether any later bug comes from the Python change or from the new feature.

**Step 3 — Install MediaPipe and get the Face Landmarker model**
`pip install mediapipe`. The model isn't bundled in the pip package — it's downloaded once and cached locally. As of this research, Google hosts a working model bundle at `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`, but confirm the current path from MediaPipe's own Face Landmarker documentation at build time, since model hosting paths can move.

**Step 4 — Build the landmark extraction function**
A function taking a frame + a face region (from the existing tracker) and returning the 468 landmark points for that face, using `FaceLandmarker` in VIDEO or LIVE_STREAM running mode — not IMAGE mode, which is for single static photos and won't perform well called every frame.

**Step 5 — Build the pattern renderer**
A drawing function taking the landmarks + a recognition status (`recognized`/`unrecognized`) that draws the contours-style pattern in the chosen color, or draws nothing if recognized. Keep this a pure rendering function — it makes no recognition decisions itself, only reads the status it's given.

**Step 6 — Wire it into the existing frame loop**
Call the landmark extraction + renderer once per tracked face, alongside (not replacing) the existing YuNet/SFace calls. Apply the every-2nd-or-3rd-frame throttling from the performance note above.

**Step 7 — Test both states explicitly**
Verify the pattern appears and persists on an unregistered face across multiple frames, then verify it disappears the moment that same face is registered/recognized. Test this as an actual before/after, not just a visual glance at one state.
