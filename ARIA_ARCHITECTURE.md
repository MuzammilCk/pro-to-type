# ARIA — Architecture & Design Rationale

## The core principle: separate seeing from thinking

The instinct behind this question was correct: if the LLM has to process visual frames directly, it becomes heavy, slow, and expensive. The fix — standard practice across CV+agent systems, and exactly the pattern the competition's "Agentic Vision" track rewards — is to keep OpenCV doing 100% of the seeing, and only hand the LLM small, structured facts about what it saw.

OpenCV never sends pixels to the LLM. It sends events like:

```json
{"event": "unknown_face", "track_id": 7, "confidence": 0.81, "seen_before": false}
{"event": "known_face", "name": "Muzammil", "track_id": 3, "last_seen_minutes_ago": 130}
```

The LLM only ever reasons over these small JSON events plus the running conversation transcript — never raw video. This is the single biggest lever for keeping the whole thing workable on a CPU-only laptop.

## Component map

1. **Perception Layer** — OpenCV 5, runs continuously, cheap
2. **Presence / State Manager** — turns raw per-frame detections into meaningful, de-duplicated events
3. **Conversation Orchestrator** — the "brain"; receives events + user speech, decides what ARIA says
4. **Audio I/O** — STT in, TTS out, with interrupt handling
5. **Registered-Faces Store** — local embeddings database
6. **AWS Layer** — where the actually expensive/valuable cloud work happens

### 1. Perception Layer

Use OpenCV 5's built-in face module: `cv.FaceDetectorYN` for detection (the YuNet model, ONNX, ~338KB, fast enough for CPU) and `cv.FaceRecognizerSF` for recognition. Both classes are present in OpenCV 5.0's own documentation with the same Python API as 4.x, so this isn't a guess — it's a direct carryover.

- Run detection on a **downscaled frame** (e.g. 50%) — detection is cheap, recognition is the expensive step.
- Only run recognition when a **new track** appears or an existing track hasn't been matched in a while — not on every frame for every face.

### 2. Presence / State Manager

- Maintain a table of currently-visible track IDs → `{name-or-unknown, first_seen, last_seen, greeted: bool}`
- Emit exactly **one event per state transition** — `person_entered`, `person_recognized`, `person_unrecognized`, `person_left` — never one event per frame
- Cooldown: a person already greeted in the last N minutes shouldn't re-trigger the same event repeatedly

### 3. Conversation Orchestrator

This is a state machine: `idle → someone enters → engage → conversing → they leave / conversation ends → idle`.

- On `person_recognized` → a personalized opener drawing on who they are
- On `person_unrecognized` → a friendly opener asking who they are — **not** an accusatory "intruder" framing. This both demos better and is the more honest posture for "hasn't met this person yet" rather than "detected a threat."
- Once a conversation is active, ARIA should respond to open-ended speech from **anyone present**, not just be gated by recognition status. Recognition decides how a conversation *starts*; it shouldn't be a gate on whether ARIA talks at all.
- This is the natural home for **AWS Bedrock**: send the current event + short conversation history, get back what ARIA says next. A cheap model tier (e.g. Amazon Nova Micro) costs a small fraction of a cent per exchange, so a standard AWS sign-up credit comfortably covers weeks of hackathon-scale conversation — and it means the actual "thinking" of the companion literally executes on AWS, which is a stronger, more literal "meaningful AWS component" than a database sitting on the side logging things.

### 4. Audio I/O — the honest version of "full duplex"

True full duplex needs acoustic echo cancellation (so ARIA doesn't hear its own voice as input) plus low-latency streaming ASR — genuinely hard, and not realistic to fully solve from scratch on this timeline/hardware/budget.

What **is** realistic: **VAD-based interruption.** Run a lightweight voice-activity detector (e.g. Silero VAD — small, CPU-friendly, pip-installable) continuously on the mic input while ARIA is speaking. The instant it detects the person started talking, stop TTS playback immediately and switch to listening. This isn't literally simultaneous processing, but the *perceived* experience is what actually reads as natural conversation to a human — and to a judge watching a demo video.

Be precise in the technical report about which of these was built: "interruptible turn-taking via VAD" is a true and still-impressive claim. "Full duplex" is a claim a judge who has built voice products may probe on.

### 5. Registered-Faces Store

Local, simple: SQLite or a JSON file mapping name → face embedding (+ metadata: registered when, by whom).

Decide explicitly which behavior is wanted for unrecognized people, since it changes both the privacy story and the code:
- **(a) Log-only** — flag the encounter, store track ID + timestamp + snapshot, but don't grant "known" status unless a human approves it later
- **(b) Auto-enroll** — after a friendly conversation where they give their name, offer to remember them next time

(a) is the safer default for a hackathon demo and a cleaner tech-report story.

### 6. AWS Layer

- **Bedrock** — conversation generation (the "brain," see above)
- **DynamoDB** — encounter log: track ID, name-or-unknown, timestamp, brief conversation summary — this becomes evaluation evidence for the final report
- **S3** — the flagged snapshot image for any unrecognized-person event — demo material and judge-accessible evidence in one
- **Optional: SNS** — a notification when someone new is logged; a cheap, real "act" step beyond just talking

## Responsible-use notes

*(This isn't only an ethics footnote — "responsible operation" is explicitly part of the graded rubric.)*

- Get explicit consent from anyone whose face is registered as "known" — the team plus anyone who agrees, never captured without their knowledge
- For demo footage of the "unrecognized visitor" scenario, use a **consenting person playing that role** rather than filming actual strangers unaware — cleaner consent story and a more controllable demo take
- State plainly in the technical report what happens to an unrecognized person's data (logged locally/in DynamoDB, not shared externally, retention if any)

## Efficiency notes for low-spec hardware

- Downscale before detection; only run recognition on new/lost tracks
- Batch/limit Bedrock calls to meaningful conversational turns, not every video frame
- Keep the real-time perception + audio loop entirely local; only route to AWS the parts that can tolerate a network round-trip (LLM reasoning, logging)
