# ARIA Agent Roadmap

**Status:** Active — Phase 1 complete; Phase 2 ready to begin
**Supersedes:** `ARIA_BUILD_TASKS.md` (its tasks are complete — keep it for history, don't add new tasks there)
**Extends:** `ARIA_ARCHITECTURE.md`, and reworks the source proposal (the 25-section "Full Companion Agent Architecture" doc) into something buildable incrementally by one person
**Companion files:** `ARIA_AGENT_PROGRESS.md` (living status — update after every phase), `ARIA_AGENT_PLAYBOOK.md` (how to actually run this with a coding agent)

## Why this plan differs from the original proposal

The original proposal is architecturally reasonable but was scoped for a funded team over months, not a solo open-ended project. It also diagnosed ARIA's problems in purely structural terms — "the LLM doesn't own a stable world representation," "memory is retrieved manually" — without engaging with the fact that the actual reported failures (ARIA never responding; recognition showing 0%) turned out to be two one-line bugs (`MicVAD.start()` never called; a `score` key dropped from a dict), already fixed as of Phase 0. A big-bang rewrite driven by structural critique, on top of a codebase that has already produced confidently-wrong documentation twice (a fabricated-sounding model-filename claim, and a since-corrected but initially-wrong claim that a real model file didn't exist), is a bad bet. More moving parts make that exact failure mode — confident, untested claims — easier to hide, not harder.

This plan keeps the destination but changes the route:
- **Smallest, highest-value fixes first.** Several phases below are cheap and fix bugs already found, not speculative architecture.
- **Build on what exists, don't discard it.** `vision_events`/action queues, `ConversationManager`, `VisionContext`, and `ProactiveEngine` already cover a meaningful fraction of what the proposal describes as missing. Phases 2, 3, and 5 formalize these — they don't replace them.
- **Every phase ends with a working, demoable ARIA.** No phase may leave the system half-migrated. If a phase can't ship in a working state, it isn't scoped correctly — split it further.
- **Nothing is "done" without a runnable proof.** Every phase has a Definition of Done that is a command to run or a specific behavior to observe live — not a diff that "looks right."
- **Scope is a hard boundary per phase.** Each phase lists what's explicitly out of scope. A diff touching files outside a phase's deliverables should be questioned, not accepted, even if it looks like an improvement.

## Ground rules (apply to every phase, not repeated below)

1. One phase at a time. Don't start phase N+1 until phase N's Definition of Done has actually been run and passed.
2. Every phase is its own commit(s), with a message stating what was verified and how — mirror the `259cbcd` "Verified-fix round" commit style, not the untitled `8757956` one.
3. If a phase reveals that an earlier "verified fact" was wrong, or an assumption here doesn't hold, stop and log it in `ARIA_AGENT_PROGRESS.md`'s corrections ledger before continuing.
4. When in doubt about scope, do less. It's always cheaper to open a follow-up phase than to unwind an overreaching one.

---

## Phase 0 — Baseline (complete, verified on hardware)

**Done:** `mic.start()` wired in; `score` key passed through to the web UI; `anon_<date>` load fallback implemented (`259cbcd`). Face-tracker cost-matrix assignment, enrollment quality gates, and revocation propagation added, backed by 16 new security tests (`8757956`).

**Outstanding gate before Phase 1 can start:**
- [x] `python3 -m pytest -v` on your actual machine, real model files present, fully green (78 passed).
- [x] One real, live, end-to-end conversation: say something, get a relevant spoken reply, confirm the face box shows a real (non-zero) percentage (verified live: "tell me my name" -> recognized MuzammilCK + memory response; "how are you" -> spoken reply; 66-85% face box).

Do not start Phase 1 until both are checked and logged in `ARIA_AGENT_PROGRESS.md`.

---

## Phase 1 — Close the two remaining structural gaps

**Goal:** fix what would otherwise quietly corrupt data collected in every later phase.

**Deliverables:**
- `context_memory.py`: gate `save()` behind an actual-change check — don't persist on every turn regardless of content.
- `context_memory.py`: stop sharing one `anon_<date>.json` across every unrecognized visitor in a day. **Default:** make unknown-identity memory session-scoped and non-persistent (cleared when the track ends, never written to disk) instead of patching the shared-file scheme — you don't have consent to build persistent profiles of unenrolled people, and it sidesteps the multi-stranger collision problem entirely. If you want persistence for strangers later, that's a deliberate, separate decision — log it as one.
- `agent.py` / `VisionContext`: introduce a small structured `WorldState`-ish object (people, objects, conversation flags) that `context_text()` renders from, rather than building the string directly. No external behavior change — this is the seed Phase 2 and 3 build on.

**Out of scope:** `tracker.py`, `voice.py`, the LLM prompt content, the web UI.

**Definition of Done:**
- [x] New test: calling `think()`/`stream_response()` twice with no new information does not call `save()` a second time (`test_phase1.py`).
- [x] New test: two different unrecognized visitors in one session never see each other's learned facts (`test_phase1.py`).
- [x] Full suite green (78/78 passing).

---

## Phase 2 — Name the events that already exist

**Goal:** formalize the signals already flowing through your vision/action queues into named events, so later phases react to defined signals instead of re-deriving "what changed" each time.

**Deliverables:** a small `events.py` — an `Event` dataclass and an enum covering only events with a real producer and consumer today: `PERSON_ENTERED`, `PERSON_LEFT`, `IDENTITY_CONFIRMED`, `USER_UTTERANCE`, `AGENT_STARTED_SPEAKING`, `AGENT_STOPPED_SPEAKING`. Wire the existing vision loop and voice session to emit these into the queues you already have.

**Out of scope:** a generic pub/sub bus, subscriber registration, or any event type without a real consumer yet. Resist adding the full 15-event list from the original proposal "for later" — add an event when a phase actually needs to react to it, not before.

**Definition of Done:** a 2-minute live session with event names logged to console; `PERSON_ENTERED` fires once per arrival (not once per frame), `USER_UTTERANCE` fires once per turn.

---

## Phase 3 — Make dialogue state explicit

**Goal:** `ConversationManager` already is a state machine; make its states an explicit enum with a transition table, and add a distinct `AgentState` (mode, active_person, active_goal, last_activity) separate from the `WorldState` from Phase 1 — the "world state != agent state" distinction the original proposal identifies, applied to what already exists.

**Deliverables:** explicit `DialogueState` enum, transition function/table, `AgentState` dataclass wired into `VoiceSession`/`ConversationManager`.

**Out of scope:** the LLM prompt/reasoning logic itself.

**Definition of Done:** existing conversation tests pass unmodified; one new test asserts an illegal transition (e.g. `IDLE → ENROLLING` with no person present) is rejected or logged, not silently allowed.

---

## Phase 4 — Real memory: working / semantic / episodic

**Goal:** the single highest-value phase for "feels like a companion." Replace the flat `PersonaGraph` with three layers plus a selective retriever. This is where episodic recall ("last Tuesday you mentioned...") actually starts happening.

**Deliverables:**
- `memory/working.py` — last N turns, current topic, pending question (session-scoped, matching Phase 1's stranger-memory scope).
- `memory/semantic.py` — stable facts about a known, enrolled person. Absorbs what the original proposal calls "procedural" memory (interaction preferences) as fields here, not a 4th subsystem — add a real separate procedural layer later only if this genuinely doesn't hold up.
- `memory/episodic.py` — dated, discrete interaction records.
- `memory/retrieval.py` — `retrieve(query, person, limit=5)`: recency + person-scope filtering is enough to start. No embeddings/vector search yet — real follow-up, not part of this phase.
- The Phase 1 write-gate formalizes into `memory/writer.py`: a gate deciding whether a candidate fact is persistent/useful/certain enough to store.

**Out of scope:** vector/semantic-similarity retrieval; a formal 4th procedural tier; away/return session summarization (fine as an optional stretch add-on here, not required for Done).

**Definition of Done:** two real sessions with your enrolled face, app restart in between; ARIA references something specific from session 1, unprompted, in session 2 — paste the actual exchange into the progress log.

---

## Phase 5 — Policy layer

**Goal:** formalize `ProactiveEngine`'s existing rate-limit/relevance gating into explicit, independently-testable policy functions (`should_greet`, `should_interrupt`, `should_follow_up`) rather than leaving these calls to the LLM or bare timers.

**Deliverables:** `policy.py` — pure functions over `WorldState`/`AgentState`/recent memory, returning a decision. No LLM call inside them.

**Definition of Done:** unit tests per function, fixed inputs → expected output, no network/LLM call required to run them.

---

## Phase 6 — Small tool/capability layer

**Goal:** start the "LLM requests, system executes" pattern with tools mapping to things already happening implicitly — not the full 15-tool list from the original proposal.

**Deliverables:** exactly four tools — `memory.search`, `memory.store`, `vision.get_state`, `speech.speak` — plus a thin dispatcher the LLM's tool-call routes through.

**Out of scope:** a general plugin/registration framework, permission contexts, or any tool without an immediate real use.

**Definition of Done:** one scripted conversation where a tool call is logged and its actual effect matches what was logged.

---

## Phase 7 — Tasks (optional — gate this deliberately)

Only start if you actually want ARIA to do things beyond real-time conversation — reminders, "watch the door," background monitoring. If the goal is purely a great real-time companion, leave this unstarted indefinitely; it solves a problem you haven't reported having.

---

## Phase 8 — Repository restructuring (last, mechanical only)

Move existing modules into `core/`, `perception/`, `memory/`, `cognition/`, `actions/`, `interaction/` per the original proposal's layout — once the shape of the system has stabilized through phases 1–6/7. Its own dedicated session, no feature changes in the same commit, tests green before and after.
