# ARIA Agent Playbook

How to actually run `ARIA_AGENT_ROADMAP.md` with a coding agent (Codebuff, Claude Code, Cursor, whatever you're using) without it turning into another big-bang rewrite.

## The loop, every phase

1. Open `ARIA_AGENT_PROGRESS.md`. Confirm the previous phase's Definition of Done is actually checked off with real output, not just "done."
2. Give the agent that phase's prompt (below) — one phase, one prompt, one session.
3. **Read the diff yourself before accepting it.** If it touches files the phase didn't list, stop and ask why before taking it — don't just accept an "improvement." This already happened once: the security-hardening commit went well beyond what was asked, with no verification note in the message. It turned out to be good work, but you got lucky reviewing it after the fact rather than catching scope before it landed.
4. Run the Definition of Done yourself, or have the agent run it and paste real output — never take "this should work now" as sufficient.
5. Once verified, use the update prompt (below) to have the agent append a properly-formatted entry to `ARIA_AGENT_PROGRESS.md`.
6. Only then move to the next phase's prompt.

Never give the agent more than one phase's prompt in a session. If you're tempted to say "and also do phase N+1 while you're at it," that's exactly the instinct that produced the 2,300-line unscoped commit — don't.

---

## Starter prompt (use once, to set the agent up on this system)

```
Read ARIA_AGENT_ROADMAP.md and ARIA_AGENT_PROGRESS.md in full before doing
anything else.

ARIA_AGENT_PROGRESS.md is the source of truth for what's actually done and
verified — not commit history, not your own memory of past sessions.

From now on we work one phase at a time, exactly as scoped in the Roadmap.
Do not touch files outside the current phase's listed deliverables, even if
you spot something else worth fixing — note it instead and I'll decide
whether it becomes its own phase.

Do not mark a phase's Definition of Done as satisfied without actually
running it and showing me the real output. "Should work" is not done.

Confirm you've read both files and tell me which phase is current before
writing any code.
```

## Per-phase prompts

Each assumes the starter prompt already ran earlier. Paste the one matching the current phase from `ARIA_AGENT_PROGRESS.md`.

**Phase 1:**
```
Implement Phase 1 from ARIA_AGENT_ROADMAP.md — the memory write-gate, the
stranger-memory scoping fix, and the WorldState-lite refactor of
VisionContext. Nothing outside those three deliverables.

Match the roadmap's recommendation for stranger memory (session-scoped,
non-persistent) unless you have a specific reason not to — if so, stop and
tell me before implementing something different.

When done, run the full test suite and the two new tests the phase calls
for, and show me the actual output. Don't tell me it's done — show me it
passing.
```

**Phase 2:**
```
Implement Phase 2 from ARIA_AGENT_ROADMAP.md — the events.py module and
wiring for exactly the six event types it lists. Do not add any event type
not already listed, even ones that seem obviously useful — if you think
one's missing, tell me, don't add it.

Wire it into the existing vision_events/action-event queues — this is
formalizing what's there, not replacing the threading model.

When done, show me the console log from a real 2-minute session confirming
PERSON_ENTERED fires once per arrival and USER_UTTERANCE fires once per
turn, not once per frame/poll.
```

**Phase 3:**
```
Implement Phase 3 from ARIA_AGENT_ROADMAP.md — an explicit DialogueState
enum + transition table for ConversationManager, and a separate AgentState
dataclass. Don't touch the LLM prompt or reasoning logic.

Run the existing conversation tests unmodified and confirm they still pass,
then add the illegal-transition test the phase calls for and show me it
passing.
```

**Phase 4:**
```
Implement Phase 4 from ARIA_AGENT_ROADMAP.md — working/semantic/episodic
memory layers and the selective retriever. Fold what the original proposal
calls "procedural" memory into semantic as fields, don't build a fourth
subsystem. No vector search — recency + person-scope filtering only, per
the roadmap.

This phase's Definition of Done needs a real two-session test with an app
restart in between, using my enrolled face — I'll run that part myself and
paste you the transcript once you say the code's ready.
```

**Phase 5:**
```
Implement Phase 5 from ARIA_AGENT_ROADMAP.md — policy.py with
should_greet / should_interrupt / should_follow_up as pure functions over
WorldState/AgentState/recent memory. No LLM or network call inside any of
them — that's the whole point, show me the unit tests proving it.
```

**Phase 6:**
```
Implement Phase 6 from ARIA_AGENT_ROADMAP.md — exactly the four tools it
lists (memory.search, memory.store, vision.get_state, speech.speak) and a
thin dispatcher. Do not build a general plugin/registration framework or
add tools beyond these four.

Show me a logged tool call from a real conversation and confirm its logged
effect matches what actually happened.
```

**Phase 7 / Phase 8:**
```
[Only start these if you've deliberately decided to — see the Roadmap's
gating notes for each. When you do, write the prompt the same way as
above: name the phase, restate its exact deliverables and exclusions from
the Roadmap, and require real verification output before it counts as
done.]
```

**Phase 9:**
```
Proceed with Phase 9 of the upgrade: create perception/inspection.py containing a
standalone class VisionInspectionEngine with stateless, CPU-optimized OpenCV 5 routines:
crop_roi(frame, bbox), inspect_color_hsv(roi, lower_hsv, upper_hsv), analyze_geometry(roi),
and measure_optical_flow(prev_roi, curr_roi). All routines must handle empty/out-of-bounds crops
safely and execute in under 5 ms on CPU. Do not touch agent.py, voice.py, or run.py yet.
Create a hermetic unit test file test_inspection.py covering all methods with numpy frames,
run pytest -v, and show the passing output.
```

**Phase 10:**
```
Implement Phase 10: Native Tool Schemas & Local LLM Adapter.
Update core/tools.py to replace regex bracket parsing with OpenAI-compliant function declarations
(tools=[{"type": "function", ...}]). Update cognition/llm_interface.py to accept tools payloads,
parse native message.tool_calls, and support local Ollama / OpenAI-compatible endpoints.
Add unit tests in test_phase10_tools_llm.py proving schema validation and mock tool-calling execution.
```

**Phase 11:**
```
Implement Phase 11: Recursive ReAct Cognitive Engine.
Rebuild cognition/reasoner.py with AgenticReasoner.evaluate_scene(), executing a closed
while finish_reason == "tool_calls" loop capped at 2 turns. Return OpenCV inspection observations
as role="tool" messages to the model. Emit non-blocking audio cues to prevent UI stalls.
Verify multi-turn hypothesis testing with hermetic tests.
```

**Phase 12:**
```
Implement Phase 12: Autonomous Loops, Visual Anomaly Trigger & Local Actuation.
Add a non-blocking Snapshot Ring Buffer and VISUAL_ANOMALY_DETECTED trigger in run.py _vision_loop().
Update actions/alerter.py to save high-res crops to ./evidence/incident_<ts>.jpg.
Update memory/writer.py to append ReAct audit traces to ./evidence/audit_log.jsonl.
Verify end-to-end autonomous visual triggers without microphone input.
```

## Update prompt (after every verified phase)

```
Phase <N> is verified — here's the output: <paste it>.

Append an entry to ARIA_AGENT_PROGRESS.md's "Completed phases" section using
the template at the bottom of that file. Use the real commit hash(es) and
the actual verification output I just gave you, not a paraphrase. Then
update the "Current status" line at the top of the file to point at the
next phase.

If anything you learned this phase contradicts something in the Verified
facts & corrections ledger or the Roadmap itself, add a new ledger entry —
don't silently work around it.
```

## Watch for, given what's already happened in this repo

- **Confident, untested claims in docs.** This has happened twice already (the yunet filename, both directions). Any doc claim that sounds oddly specific — "requires X because Y" — deserves a real check, not trust.
- **Scope creep dressed as improvement.** The security-hardening commit was good work but happened unscoped and unverified-on-paper. Good outcome, bad process — don't rely on that going well a second time.
- **"Should work" as a substitute for "does work."** Every phase in the Roadmap has a Definition of Done that's a command or an observed behavior, not a code review. Hold that line every time, especially when the agent sounds confident.
