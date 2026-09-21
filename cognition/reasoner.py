"""Cognitive ReAct Reasoner — iterative hypothesis testing and active visual investigation (Phase 11).

Replaces legacy static heuristics with an autonomous ReAct loop:
  Perception Trigger -> Thought -> Tool Execution on Camera Buffer -> Observation -> Final Decision

Features:
1. Multi-turn closed ReAct loop (while finish_reason == "tool_calls")
2. Hard-capped at 2 iterations for real-time responsiveness (<2s total)
3. Non-blocking audio cues on initial tool invocation
4. Structured decision synthesis: CLEAR, WARN, HALT, ENROLL
5. Full backward-compatibility with legacy Reasoner.evaluate() for existing tests/pipelines.
"""
from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable
import numpy as np

from perception.detector import Detection
from core.tools import ToolDispatcher, get_openai_tool_definitions, ToolResult
from cognition.llm_interface import LLMClient, LLMResponse, LocalFallbackLLM


class Reasoner:
    """Cognitive reasoner supporting both legacy detection evaluations and agentic ReAct loops."""

    WATCHED_LABELS = {"person", "cat", "dog"}
    AUTHORIZED = set()

    def __init__(
        self,
        face_engine: Any = None,
        llm: LLMClient | None = None,
        dispatcher: ToolDispatcher | None = None,
    ):
        self.face_engine = face_engine
        self.llm = llm or LocalFallbackLLM()
        self.dispatcher = dispatcher

    def set_authorized(self, names: set[str]):
        """Update the set of authorized identities."""
        self.AUTHORIZED = names

    # ------------------------------------------------------------------
    # Legacy evaluation path (retained for backward compatibility)
    # ------------------------------------------------------------------

    def evaluate(self, detections: list[Detection], frame: np.ndarray | None = None) -> dict[str, Any]:
        """Evaluate scene detections and face authorizations via heuristic checks."""
        person_dets = [
            d for d in detections
            if getattr(d, "label", "") in self.WATCHED_LABELS and getattr(d, "confidence", 0.0) > 0.7
        ]

        face_results = []
        threats = []

        if self.face_engine is not None and frame is not None:
            faces = self.face_engine.detect(frame)
            for face in faces:
                x, y, w, h = [int(v) for v in face[:4]]
                landmarks = face[4:14].astype(int).tolist()
                name, dist, meta = self.face_engine.identify(frame, face)
                authorized = name in self.AUTHORIZED
                face_results.append({
                    "identity": name,
                    "authorized": authorized,
                    "distance": round(dist, 3),
                    "face_bbox": (x, y, w, h),
                    "landmarks": landmarks,
                    "quality": meta.get("quality", 0),
                })
                if not authorized:
                    threats.append({
                        "identity": name,
                        "distance": round(dist, 3),
                        "face_bbox": (x, y, w, h),
                    })

        should_act = len(threats) > 0
        return {
            "should_act": should_act,
            "threats": threats,
            "face_results": face_results,
            "total": len(detections),
            "reason": "unauthorized_person" if threats else "authorized_or_no_person",
        }

    # ------------------------------------------------------------------
    # Agentic ReAct Evaluation Core (Phase 11)
    # ------------------------------------------------------------------

    def evaluate_scene(
        self,
        trigger: dict[str, Any] | str,
        frame_provider: Callable[[], np.ndarray | None] | None = None,
        llm: LLMClient | None = None,
        dispatcher: ToolDispatcher | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_iterations: int = 2,
        audio_cue_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Execute a multi-turn ReAct investigation loop against live camera observations.

        Args:
            trigger: Event or description initiating the investigation (e.g. visual anomaly or presence).
            frame_provider: Callable returning the latest camera frame.
            llm: LLMClient instance (defaults to self.llm).
            dispatcher: ToolDispatcher configured with active tools and frame provider.
            tools: List of OpenAI tool definitions (defaults to full suite).
            max_iterations: Maximum ReAct cycles before forcing decision (default 2).
            audio_cue_callback: Callback to emit immediate audio cue on first tool call.

        Returns:
            Structured decision dictionary with action, summary, trace, and evidence.
        """
        active_llm = llm or self.llm or LocalFallbackLLM()
        active_dispatcher = dispatcher or self.dispatcher or ToolDispatcher(frame_provider=frame_provider)
        if frame_provider and active_dispatcher.frame_provider is None:
            active_dispatcher.frame_provider = frame_provider

        available_tools = tools if tools is not None else get_openai_tool_definitions()

        trigger_data = trigger if isinstance(trigger, dict) else {"description": str(trigger)}
        system_prompt = (
            "You are ARIA's Autonomous Physical AI Cognitive Core.\n"
            "You have access to active computer vision micro-tools that inspect the live camera buffer.\n"
            "When an event or anomaly is reported, investigate it systematically:\n"
            "1. Formulate a hypothesis and call vision inspection tools (crop_and_enhance, inspect_color_hsv, analyze_geometry, measure_optical_flow).\n"
            "2. Analyze the returned observations.\n"
            "3. Formulate your final operational decision in this exact format:\n"
            "DECISION: <CLEAR | WARN | HALT | ENROLL> - <reasoning>\n\n"
            "- CLEAR: Situation verified safe / authorized / normal.\n"
            "- WARN: Potential hazard, missing tag, or minor anomaly observed.\n"
            "- HALT: Critical hazard, unauthorized access, or safety violation.\n"
            "- ENROLL: Known person requesting or requiring identity enrollment."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Visual scene trigger: {json.dumps(trigger_data)}"},
        ]

        trace: list[dict[str, Any]] = []
        turn = 0
        final_text = ""

        while turn < max_iterations:
            turn += 1
            resp: LLMResponse = active_llm.chat_complete(messages, tools=available_tools)

            if not resp.has_tool_calls or resp.finish_reason != "tool_calls":
                # Direct synthesis reached without further tool calls
                final_text = resp.content
                break

            # Emitted tool calls: trigger immediate non-blocking audio cue on first turn
            if audio_cue_callback and turn == 1:
                try:
                    audio_cue_callback("Inspecting workspace visual evidence.")
                except Exception:
                    pass

            # 1. Append assistant tool_calls message
            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": resp.tool_calls,
            })

            # 2. Execute requested tool calls against live frame
            turn_results: list[ToolResult] = []
            for tc in resp.tool_calls:
                res = active_dispatcher.execute_tool_call(tc)
                turn_results.append(res)
                # Append tool observation
                messages.append(res.to_message())

            trace.append({
                "iteration": turn,
                "thought": resp.content,
                "tool_calls": resp.tool_calls,
                "observations": [r.to_dict() for r in turn_results],
            })

        else:
            # Reached max iteration limit without termination: force final synthesis turn without tools
            forced_resp = active_llm.chat_complete(messages, tools=None)
            final_text = forced_resp.content

        # Parse structured decision
        decision = self._parse_decision(final_text, trace)

        all_observations = [obs for step in trace for obs in step["observations"]]
        should_act = decision in ("WARN", "HALT", "ENROLL")

        return {
            "decision": decision,
            "summary": final_text,
            "iterations": turn,
            "trace": trace,
            "evidence": all_observations,
            "should_act": should_act,
        }

    @staticmethod
    def _parse_decision(text: str, trace: list[dict[str, Any]]) -> str:
        """Extract or infer the categorical decision from response text or trace evidence."""
        if text:
            m = re.search(r"DECISION:\s*(CLEAR|WARN|HALT|ENROLL)", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()

            low = text.lower()
            if any(w in low for w in ("halt", "danger", "critical", "stop", "hazard")):
                return "HALT"
            if any(w in low for w in ("warn", "caution", "anomaly", "suspicious", "missing")):
                return "WARN"
            if any(w in low for w in ("enroll", "register")):
                return "ENROLL"
            if any(w in low for w in ("clear", "normal", "safe", "verified", "authorized")):
                return "CLEAR"

        # Fallback based on tool observation anomalies
        for step in trace:
            for obs in step.get("observations", []):
                out = obs.get("output", {})
                if isinstance(out, dict):
                    if out.get("detected") is False and "color" in obs.get("tool", ""):
                        return "WARN"
                    if out.get("is_moving") is True:
                        return "WARN"

        return "CLEAR"


# Alias for explicit agentic naming
AgenticReasoner = Reasoner
