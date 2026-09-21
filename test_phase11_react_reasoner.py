"""Unit and integration tests for Phase 11: Recursive ReAct Cognitive Engine.

Validates:
1. Full backward compatibility with legacy Reasoner.evaluate().
2. AgenticReasoner multi-turn ReAct loop (Perception Trigger -> Thought -> Tool -> Observation -> Final Decision).
3. Tool observations successfully fed back to the LLM to inform decision.
4. Decision classification: CLEAR, WARN, HALT, ENROLL.
5. Hard iteration capping (max_iterations=2) preventing runaway loops.
6. Non-blocking audio feedback cue on first tool call.
"""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock

from perception.detector import Detection
from core.tools import ToolDispatcher, ToolResult
from cognition.reasoner import Reasoner, AgenticReasoner
from cognition.llm_interface import LLMClient, LLMResponse


class MockMultiTurnReActLLM(LLMClient):
    """Hermetic mock LLM simulating a multi-turn ReAct conversation."""

    def __init__(self, turns: list[LLMResponse]):
        self.turns = list(turns)
        self.call_history: list[list[dict]] = []

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        if self.turns:
            return self.turns.pop(0)
        return LLMResponse(content="DECISION: CLEAR - Default verification.", finish_reason="stop")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return self.chat_complete(messages, tools=tools).content

    def stream(self, messages: list[dict]):
        yield self.complete(messages)


class TestReasonerBackwardCompatibility:
    def test_legacy_evaluate(self):
        reasoner = Reasoner()
        reasoner.set_authorized({"Muzammil"})

        dets = [
            Detection(label="person", confidence=0.85, bbox=[10, 10, 50, 50]),
            Detection(label="chair", confidence=0.90, bbox=[100, 100, 50, 50]),
        ]

        decision = reasoner.evaluate(dets, frame=None)
        assert decision["total"] == 2
        assert "threats" in decision
        assert decision["should_act"] is False
        assert decision["reason"] == "authorized_or_no_person"

    def test_agentic_reasoner_alias(self):
        assert AgenticReasoner is Reasoner


class TestAgenticReActLoop:
    def test_successful_hypothesis_verification_cycle(self):
        # Frame with green patch
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[20:80, 20:80] = [0, 255, 0]

        # Turn 1: Model requests color inspection
        turn1 = LLMResponse(
            content="Formulating hypothesis: The object has a green safety indicator.",
            tool_calls=[{
                "id": "call_inspect_1",
                "type": "function",
                "function": {
                    "name": "vision.inspect_color_hsv",
                    "arguments": json.dumps({
                        "bbox": [20, 20, 60, 60],
                        "lower_hsv": [35, 50, 50],
                        "upper_hsv": [85, 255, 255],
                    }),
                },
            }],
            finish_reason="tool_calls",
        )

        # Turn 2: Model receives observation and confirms CLEAR
        turn2 = LLMResponse(
            content="Observation verified: 100% green coverage confirmed. DECISION: CLEAR - Safety indicator verified.",
            tool_calls=[],
            finish_reason="stop",
        )

        mock_llm = MockMultiTurnReActLLM([turn1, turn2])
        audio_cues = []
        reasoner = AgenticReasoner(llm=mock_llm)

        trigger = {"event": "visual_anomaly", "bbox": [20, 20, 60, 60], "label": "unverified_indicator"}
        result = reasoner.evaluate_scene(
            trigger=trigger,
            frame_provider=lambda: frame.copy(),
            audio_cue_callback=lambda cue: audio_cues.append(cue),
        )

        # 1. Verified multi-turn execution
        assert result["iterations"] == 2
        assert result["decision"] == "CLEAR"
        assert result["should_act"] is False
        assert len(result["trace"]) == 1
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["output"]["detected"] is True

        # 2. Audio cue fired on tool invocation
        assert len(audio_cues) == 1
        assert "Inspecting workspace" in audio_cues[0]

        # 3. Observation message was injected into turn 2 LLM history
        second_call_msgs = mock_llm.call_history[1]
        tool_obs_msg = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert len(tool_obs_msg) == 1
        assert tool_obs_msg[0]["tool_call_id"] == "call_inspect_1"

    def test_anomaly_triggers_warn(self):
        # Black frame (no green)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        turn1 = LLMResponse(
            content="Hypothesis: Check if green tag present.",
            tool_calls=[{
                "id": "call_color_2",
                "type": "function",
                "function": {
                    "name": "vision.inspect_color_hsv",
                    "arguments": json.dumps({"bbox": [20, 20, 60, 60], "lower_hsv": [35, 50, 50], "upper_hsv": [85, 255, 255]}),
                },
            }],
            finish_reason="tool_calls",
        )
        turn2 = LLMResponse(
            content="Color absent: 0% match. DECISION: WARN - Missing required safety tag.",
            tool_calls=[],
            finish_reason="stop",
        )

        mock_llm = MockMultiTurnReActLLM([turn1, turn2])
        reasoner = AgenticReasoner(llm=mock_llm)

        result = reasoner.evaluate_scene(
            trigger="missing_tag_check",
            frame_provider=lambda: frame.copy(),
        )

        assert result["decision"] == "WARN"
        assert result["should_act"] is True
        assert result["iterations"] == 2

    def test_hazard_triggers_halt(self):
        turn1 = LLMResponse(
            content="Hazard detected! DECISION: HALT - Moving tool encroaching safety zone.",
            tool_calls=[],
            finish_reason="stop",
        )
        mock_llm = MockMultiTurnReActLLM([turn1])
        reasoner = AgenticReasoner(llm=mock_llm)

        result = reasoner.evaluate_scene(
            trigger={"event": "perimeter_breach"},
            frame_provider=lambda: np.zeros((100, 100, 3), dtype=np.uint8),
        )

        assert result["decision"] == "HALT"
        assert result["should_act"] is True
        assert result["iterations"] == 1

    def test_hard_capping_max_iterations(self):
        """Verify runaway tool calling loops are terminated at max_iterations."""
        infinite_call = LLMResponse(
            content="Checking again...",
            tool_calls=[{
                "id": "call_inf",
                "type": "function",
                "function": {
                    "name": "vision.crop_and_enhance",
                    "arguments": json.dumps({"bbox": [0, 0, 50, 50]}),
                },
            }],
            finish_reason="tool_calls",
        )

        # Always returns tool call, forcing iteration cap
        mock_llm = MockMultiTurnReActLLM([infinite_call, infinite_call, infinite_call])
        reasoner = AgenticReasoner(llm=mock_llm)

        result = reasoner.evaluate_scene(
            trigger="infinite_trigger",
            frame_provider=lambda: np.zeros((100, 100, 3), dtype=np.uint8),
            max_iterations=2,
        )

        # Loop stopped at exactly 2 iterations
        assert result["iterations"] == 2
        assert len(result["trace"]) == 2
