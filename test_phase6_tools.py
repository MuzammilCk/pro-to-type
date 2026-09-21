"""Unit and integration tests for Phase 6 — Small Tool / Capability Layer.

Tests:
1. Four tools defined in TOOL_SCHEMAS: memory.search, memory.store, vision.get_state, speech.speak.
2. memory.search retrieves facts and episodes scoped to an enrolled person.
3. memory.store persists a fact for an enrolled person to memory and disk.
4. memory.store rejects unenrolled strangers / unknown visitors via privacy barrier.
5. vision.get_state returns structured camera context (objects, faces, summary, lighting).
6. speech.speak enqueues spoken audio via voice synthesizer.
7. ToolDispatcher parses bracket syntax [TOOL: name(args)] and json blocks.
8. ToolDispatcher logs invocations in call_log with tool, arguments, success, and output.
9. strip_tool_calls cleans tool syntax from spoken dialogue.
10. Definition of Done: Scripted conversation where LLM turn requests tool execution,
    invocation is logged, and system state is verified to have changed accordingly.
"""
import pytest
import os
import json
import shutil
from unittest.mock import MagicMock

from core.tools import (
    ToolResult,
    TOOL_SCHEMAS,
    ToolDispatcher,
    format_tools_prompt,
    strip_tool_calls,
)
from memory.context_memory import PersonMemory
from cognition.agent import VisionAgent, VisionContext, VoiceSession


@pytest.fixture
def clean_memory_dir(tmp_path, monkeypatch):
    """Hermetic memory directory for tests."""
    mem_dir = tmp_path / "memory_store"
    mem_dir.mkdir(parents=True, exist_ok=True)
    import memory.semantic
    import memory.episodic
    import memory.context_memory
    monkeypatch.setattr(memory.semantic, "PERSONA_DIR", str(mem_dir))
    monkeypatch.setattr(memory.episodic, "PERSONA_DIR", str(mem_dir))
    monkeypatch.setattr(memory.context_memory, "PERSONA_DIR", str(mem_dir))
    return str(mem_dir)


# ----------------------------------------------------------------------
# 1. Tool schemas & declarations
# ----------------------------------------------------------------------

def test_tool_schemas_exact_four():
    """Verify exactly four specified tools exist in TOOL_SCHEMAS."""
    expected = {"memory.search", "memory.store", "vision.get_state", "speech.speak"}
    assert set(TOOL_SCHEMAS.keys()) == expected
    for name in expected:
        schema = TOOL_SCHEMAS[name]
        assert schema["name"] == name
        assert "description" in schema
        assert "parameters" in schema


def test_format_tools_prompt():
    prompt = format_tools_prompt()
    assert "AVAILABLE TOOLS:" in prompt
    assert "memory.search" in prompt
    assert "memory.store" in prompt
    assert "vision.get_state" in prompt
    assert "speech.speak" in prompt


# ----------------------------------------------------------------------
# 2. memory.search
# ----------------------------------------------------------------------

def test_memory_search_enrolled_person(clean_memory_dir):
    agent = VisionAgent()
    mem = PersonMemory(identity="Alice")
    mem.add_fact("role", "Lead Engineer")
    mem.add_fact("framework", "OpenCV")
    mem.save()
    agent.memory["Alice"] = mem

    dispatcher = ToolDispatcher(agent=agent)
    res = dispatcher.execute("memory.search", {"query": "OpenCV", "person": "Alice"})

    assert res.success is True
    assert res.tool == "memory.search"
    assert res.output["person"] == "Alice"
    assert "framework" in res.output["facts"]
    assert res.output["facts"]["framework"] == "OpenCV"


# ----------------------------------------------------------------------
# 3. memory.store — enrolled person
# ----------------------------------------------------------------------

def test_memory_store_enrolled_person(clean_memory_dir):
    agent = VisionAgent()
    mem = PersonMemory(identity="Bob")
    mem.save()
    agent.memory["Bob"] = mem

    dispatcher = ToolDispatcher(agent=agent)
    res = dispatcher.execute("memory.store", {"key": "hobby", "value": "robotics", "person": "Bob"})

    assert res.success is True
    assert res.output["status"] == "stored"
    assert res.output["key"] == "hobby"
    assert res.output["value"] == "robotics"

    # Verify actual underlying memory was updated and persisted
    loaded = PersonMemory.load("Bob")
    assert loaded.semantic.facts.get("hobby") == "robotics"


# ----------------------------------------------------------------------
# 4. memory.store — stranger gate barrier
# ----------------------------------------------------------------------

def test_memory_store_stranger_rejected(clean_memory_dir):
    agent = VisionAgent()
    dispatcher = ToolDispatcher(agent=agent)

    # Unknown identity must be rejected by privacy barrier
    res_unknown = dispatcher.execute("memory.store", {"key": "note", "value": "suspicious", "person": "unknown"})
    assert res_unknown.success is False
    assert "Privacy gate rejected" in (res_unknown.error or "")

    # Anonymous track identity must also be rejected
    res_anon = dispatcher.execute("memory.store", {"key": "note", "value": "visitor", "person": "anon_face_12"})
    assert res_anon.success is False
    assert "Privacy gate rejected" in (res_anon.error or "")


# ----------------------------------------------------------------------
# 5. vision.get_state
# ----------------------------------------------------------------------

def test_vision_get_state():
    ctx = VisionContext()
    ctx.summary = "1 recognized person present."

    mock_detection = MagicMock()
    mock_detection.label = "laptop"

    mock_face = {"identity": "Charlie", "authorized": True, "face_bbox": (10, 10, 50, 50)}
    ctx.update([mock_face], [mock_detection])

    dispatcher = ToolDispatcher(vision_ctx=ctx)
    res = dispatcher.execute("vision.get_state", {})

    assert res.success is True
    assert res.tool == "vision.get_state"
    assert res.output["faces_count"] == 1
    assert "laptop" in res.output["objects"]


# ----------------------------------------------------------------------
# 6. speech.speak
# ----------------------------------------------------------------------

def test_speech_speak():
    mock_voice = MagicMock()
    dispatcher = ToolDispatcher(voice=mock_voice)

    res = dispatcher.execute("speech.speak", {"text": "Hello world, systems are online."})

    assert res.success is True
    assert res.output["spoken"] == "Hello world, systems are online."
    assert mock_voice.enqueue_speech.called or mock_voice.speak_async.called or mock_voice.speak.called


# ----------------------------------------------------------------------
# 7. Parsing brackets and JSON blocks
# ----------------------------------------------------------------------

def test_tool_dispatcher_parsing():
    dispatcher = ToolDispatcher()

    bracket_text = 'Sure thing! [TOOL: memory.store({"key": "city", "value": "Seattle", "person": "Dave"})]'
    parsed = dispatcher.parse_calls(bracket_text)
    assert len(parsed) == 1
    assert parsed[0][0] == "memory.store"
    assert parsed[0][1] == {"key": "city", "value": "Seattle", "person": "Dave"}

    json_text = """
    Here is what I see:
    ```json
    {
      "tool": "vision.get_state",
      "arguments": {}
    }
    ```
    """
    parsed_json = dispatcher.parse_calls(json_text)
    assert len(parsed_json) == 1
    assert parsed_json[0][0] == "vision.get_state"


def test_strip_tool_calls():
    raw1 = 'Hello Dave! [TOOL: memory.store({"key": "editor", "value": "neovim", "person": "Dave"})] Nice to see you.'
    cleaned1 = strip_tool_calls(raw1)
    assert cleaned1 == "Hello Dave!  Nice to see you."

    raw2 = '```json\n{"tool": "vision.get_state", "arguments": {}}\n```All clear.'
    cleaned2 = strip_tool_calls(raw2)
    assert cleaned2 == "All clear."


# ----------------------------------------------------------------------
# 8. Execution logging in call_log
# ----------------------------------------------------------------------

def test_dispatcher_call_logging():
    mock_voice = MagicMock()
    dispatcher = ToolDispatcher(voice=mock_voice)

    dispatcher.execute("speech.speak", {"text": "Alert triggered."})
    dispatcher.execute("unknown.tool", {"foo": "bar"})

    assert len(dispatcher.call_log) == 2
    assert dispatcher.call_log[0]["tool"] == "speech.speak"
    assert dispatcher.call_log[0]["success"] is True

    assert dispatcher.call_log[1]["tool"] == "unknown.tool"
    assert dispatcher.call_log[1]["success"] is False


# ----------------------------------------------------------------------
# 9. Definition of Done: Scripted conversation end-to-end
# ----------------------------------------------------------------------

def test_scripted_conversation_tool_execution(clean_memory_dir):
    """A scripted LLM conversation turn emits a tool call; dispatcher executes it,

    logs it, and the actual system state matches what was logged.
    """
    agent = VisionAgent()
    mock_voice = MagicMock()
    ctx = VisionContext()
    agent.attach_tools(voice=mock_voice, vision_ctx=ctx)

    # Pre-enroll Eve
    eve_mem = PersonMemory(identity="Eve")
    eve_mem.save()
    agent.memory["Eve"] = eve_mem

    # Mock LLM completing with a dialogue turn requesting memory storage
    simulated_llm_reply = (
        'Got it Eve, I will remember that your favorite editor is VSCode! '
        '[TOOL: memory.store({"key": "favorite_editor", "value": "VSCode", "person": "Eve"})]'
    )
    agent.llm = MagicMock()
    agent.llm.available = True
    agent.llm.complete.return_value = simulated_llm_reply

    # Execute think turn
    reply, action = agent.think("Eve", {"authorized": True}, user_input="My favorite editor is VSCode")

    # 1. Verify spoken reply stripped tool syntax
    assert "[TOOL:" not in reply
    assert "Got it Eve, I will remember that your favorite editor is VSCode!" in reply

    # 2. Verify tool invocation was executed and logged
    log = agent.dispatcher.call_log
    assert len(log) == 1
    assert log[0]["tool"] == "memory.store"
    assert log[0]["success"] is True
    assert log[0]["arguments"] == {"key": "favorite_editor", "value": "VSCode", "person": "Eve"}

    # 3. Assert the actual system state changed to match what was logged
    reloaded_mem = PersonMemory.load("Eve")
    assert reloaded_mem.semantic.facts.get("favorite_editor") == "VSCode"
