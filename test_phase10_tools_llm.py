"""Unit and integration tests for Phase 10: Native Tool Schemas & Local LLM Adapter.

Validates:
1. get_openai_tool_definitions produces standard OpenAI-compliant function tool schemas.
2. ToolResult.to_message formats into standard role='tool' message dicts with tool_call_id.
3. ToolDispatcher.execute_tool_call handles native OpenAI tool-call structures.
4. ToolDispatcher dispatches vision inspection tools on live frame provider.
5. Graceful failure when camera frame is unavailable.
6. LLMResponse structured response model.
7. LocalFallbackLLM tool-call simulation for offline/hermetic multi-turn testing.
8. OllamaClient and OpenRouterClient payload formatting with tools and tool_choice.
"""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.tools import (
    ToolResult,
    TOOL_SCHEMAS,
    VISION_INSPECTION_SCHEMAS,
    ALL_TOOL_SCHEMAS,
    ToolDispatcher,
    get_openai_tool_definitions,
)
from cognition.llm_interface import (
    LLMResponse,
    LocalFallbackLLM,
    OllamaClient,
    OpenRouterClient,
)


class TestOpenAIToolSchemas:
    def test_tool_definitions_format(self):
        definitions = get_openai_tool_definitions()
        assert len(definitions) == len(ALL_TOOL_SCHEMAS)
        for tool_def in definitions:
            assert tool_def["type"] == "function"
            fn = tool_def["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_tool_definitions_filtering(self):
        subset = ["vision.crop_and_enhance", "memory.search"]
        defs = get_openai_tool_definitions(subset)
        assert len(defs) == 2
        names = {d["function"]["name"] for d in defs}
        assert names == set(subset)

    def test_tool_result_to_message(self):
        res = ToolResult(
            tool="vision.crop_and_enhance",
            success=True,
            output={"resolution": "150x150"},
            tool_call_id="call_abc123",
        )
        msg = res.to_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc123"
        parsed_content = json.loads(msg["content"])
        assert parsed_content["resolution"] == "150x150"

        err_res = ToolResult(
            tool="vision.crop_and_enhance",
            success=False,
            error="Out of bounds",
            tool_call_id="call_err",
        )
        err_msg = err_res.to_message()
        assert json.loads(err_msg["content"]) == {"error": "Out of bounds"}


class TestToolDispatcherExecution:
    def test_execute_native_tool_call_structure(self):
        mock_voice = MagicMock()
        dispatcher = ToolDispatcher(voice=mock_voice)

        native_call = {
            "id": "call_speak_42",
            "type": "function",
            "function": {
                "name": "speech.speak",
                "arguments": json.dumps({"text": "Inspection complete"}),
            },
        }

        res = dispatcher.execute_tool_call(native_call)
        assert res.success is True
        assert res.tool == "speech.speak"
        assert res.tool_call_id == "call_speak_42"
        assert res.output["spoken"] == "Inspection complete"
        assert len(dispatcher.call_log) == 1
        assert dispatcher.call_log[0]["tool"] == "speech.speak"

    def test_vision_inspection_tools_with_frame_provider(self):
        # Synthetic 200x200 BGR frame with a green box in [20:80, 20:80]
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[20:80, 20:80] = [0, 255, 0]

        frame_provider = lambda: frame.copy()
        dispatcher = ToolDispatcher(frame_provider=frame_provider)

        # 1. vision.crop_and_enhance
        res_crop = dispatcher.execute("vision.crop_and_enhance", {"bbox": [20, 20, 60, 60], "enhance": True})
        assert res_crop.success is True
        assert res_crop.output["resolution"] == "60x60"
        assert res_crop.output["enhanced"] is True

        # 2. vision.inspect_color_hsv
        res_color = dispatcher.execute(
            "vision.inspect_color_hsv",
            {"bbox": [20, 20, 60, 60], "lower_hsv": [35, 50, 50], "upper_hsv": [85, 255, 255]},
        )
        assert res_color.success is True
        assert res_color.output["detected"] is True
        assert res_color.output["coverage_ratio"] > 0.8

        # 3. vision.analyze_geometry (bbox includes contrast edges against background)
        res_geom = dispatcher.execute("vision.analyze_geometry", {"bbox": [10, 10, 80, 80]})
        assert res_geom.success is True
        assert res_geom.output["has_structure"] is True
        assert res_geom.output["contour_count"] >= 1

        # 4. vision.measure_optical_flow (first establishes baseline)
        res_flow1 = dispatcher.execute("vision.measure_optical_flow", {"bbox": [10, 10, 80, 80]})
        assert res_flow1.success is True
        assert res_flow1.output["status"] == "baseline_established"


    def test_vision_tools_fail_gracefully_when_frame_missing(self):
        dispatcher = ToolDispatcher(frame_provider=lambda: None)
        res = dispatcher.execute("vision.crop_and_enhance", {"bbox": [0, 0, 50, 50]})
        assert res.success is False
        assert "Camera frame unavailable" in res.error


class TestLLMInterfaceNativeToolCalling:
    def test_llm_response_dataclass(self):
        resp = LLMResponse(
            content="I will check the badge.",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "vision.crop_and_enhance"}}],
            finish_reason="tool_calls",
        )
        assert resp.has_tool_calls is True
        assert resp.finish_reason == "tool_calls"
        assert resp.content == "I will check the badge."

    def test_local_fallback_tool_calling_simulation(self):
        llm = LocalFallbackLLM()
        tools = get_openai_tool_definitions()

        # Prompt requesting color inspection
        msgs = [{"role": "user", "content": "Please inspect color of the badge."}]
        res = llm.chat_complete(msgs, tools=tools)

        assert res.has_tool_calls is True
        assert res.finish_reason == "tool_calls"
        assert res.tool_calls[0]["function"]["name"] == "vision.inspect_color_hsv"

        # Calling complete() returns text content
        text = llm.complete(msgs, tools=tools)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_ollama_client_payload(self):
        client = OllamaClient(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
        msgs = [{"role": "user", "content": "Hello"}]
        tools = get_openai_tool_definitions()

        payload = client._payload(msgs, tools=tools)
        assert payload["model"] == "qwen2.5:7b"
        assert payload["messages"] == msgs
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    def test_openrouter_payload(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        client = OpenRouterClient()
        msgs = [{"role": "user", "content": "Scan the area"}]
        tools = get_openai_tool_definitions()

        payload = client._payload(messages=msgs, tools=tools, stream=False)
        assert "tools" in payload
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"
        assert payload["stream"] is False
