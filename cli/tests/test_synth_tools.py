"""Tests for tool-calling dataset synthesis (-f tools)."""

import json

from troy.synth import build_tool_record, parse_tool_examples

SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
]

GOOD = {
    "user": "weather in Goa?",
    "steps": [{"think": "City given, call the tool.",
               "call": {"name": "get_weather", "arguments": {"city": "Goa"}},
               "result": {"forecast": "sunny"}}],
    "final_think": "Report the forecast.",
    "final": "Sunny in Goa.",
}


def test_parse_valid_example():
    raw = "some preamble\n" + json.dumps(GOOD) + "\ntrailing"
    assert parse_tool_examples(raw, SCHEMAS) == [GOOD]


def test_parse_rejects_unknown_tool_and_missing_required():
    bad_tool = dict(GOOD, steps=[{"think": "x", "call": {"name": "nope", "arguments": {}},
                                  "result": {}}])
    bad_args = dict(GOOD, steps=[{"think": "x", "call": {"name": "get_weather",
                                                         "arguments": {}},
                                  "result": {}}])
    raw = json.dumps(bad_tool) + "\n" + json.dumps(bad_args)
    assert parse_tool_examples(raw, SCHEMAS) == []


def test_parse_allows_no_tool_scenarios():
    ex = {"user": "hi", "steps": [], "final_think": "Just greet.", "final": "Hello!"}
    assert parse_tool_examples(json.dumps(ex), SCHEMAS) == [ex]


def test_build_record_shape_and_think():
    rec = build_tool_record(GOOD, SCHEMAS, system="test bot")
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert rec["tools"] == SCHEMAS
    assert rec["messages"][2]["content"].startswith("<think>\n")
    assert "<tool_call>" in rec["messages"][2]["content"]
    assert json.loads(rec["messages"][3]["content"]) == {"forecast": "sunny"}
    assert rec["messages"][4]["content"].endswith("Sunny in Goa.")


def test_build_record_no_think():
    rec = build_tool_record(GOOD, SCHEMAS, system="test bot", think=False)
    assert "<think>" not in rec["messages"][2]["content"]
    assert rec["messages"][2]["content"].startswith("<tool_call>")
