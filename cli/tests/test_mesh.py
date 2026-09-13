"""Tests for the mesh coordinator's queue, lease, and ingest logic."""

import json

from troy.mesh import MAX_ATTEMPTS, MeshState
from troy.synth import ingest_raw, prepare_synth

SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
]

GOOD_TOOL = {
    "user": "weather in Goa?",
    "steps": [{"think": "City given, call the tool.",
               "call": {"name": "get_weather", "arguments": {"city": "Goa"}},
               "result": {"forecast": "sunny"}}],
    "final_think": "Report the forecast.",
    "final": "Sunny in Goa.",
}


def raw_pairs(*pairs):
    return "\n".join(json.dumps({"user": u, "assistant": a}) for u, a in pairs)


def make_state(tmp_path, n=4, fmt="chat", lease_timeout=300.0):
    spec = prepare_synth(
        fmt, "being terse", None,
        tools_path=_schemas_file(tmp_path) if fmt == "tools" else None,
    )
    return MeshState(
        spec, n, tmp_path / "train.jsonl",
        max_tokens=512, temperature=0.7, lease_timeout=lease_timeout,
    )


def _schemas_file(tmp_path):
    p = tmp_path / "tools.json"
    p.write_text(json.dumps(SCHEMAS))
    return p


def test_lease_caps_batch_and_mints_task_prompts(tmp_path):
    state = make_state(tmp_path)
    items, done = state.lease("phone", 2, now=0.0)
    assert not done
    assert len(items) == 2
    assert all("being terse" in i.prompt for i in items)
    assert {i.worker for i in items} == {"phone"}


def test_expired_lease_requeues_then_drops(tmp_path):
    state = make_state(tmp_path, lease_timeout=10.0)
    first, _ = state.lease("phone", 1, now=0.0)
    item_id = first[0].id
    for round_i in range(1, MAX_ATTEMPTS):
        items, _ = state.lease("phone", 1, now=round_i * 100.0)
        assert items[0].id == item_id
        assert items[0].attempts == round_i
    # final expiry drops it; the next lease gets a freshly minted item
    items, _ = state.lease("phone", 1, now=MAX_ATTEMPTS * 100.0)
    assert items[0].id != item_id
    assert state.snapshot()["queue"]["dropped"] == 1


def test_ingest_appends_dedupes_and_finishes(tmp_path):
    state = make_state(tmp_path, n=3)
    items, _ = state.lease("phone", 1, now=0.0)
    reply = state.ingest("phone", items[0].id, raw_pairs(("a?", "A."), ("b?", "B.")))
    assert reply == {"accepted": 2, "duplicates": 0, "records": 2,
                     "target": 3, "done": False}

    # duplicate of "a?" plus enough new records to overshoot the target
    reply = state.ingest(
        "phone", "w-999999",  # unknown/expired id still ingests
        raw_pairs(("A?", "again"), ("c?", "C."), ("d?", "D.")),
    )
    assert reply["duplicates"] == 1
    assert reply["accepted"] == 1  # surplus truncated at target
    assert reply["done"] is True

    lines = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["messages"][0] == {"role": "user", "content": "a?"}

    _, done = state.lease("phone", 1, now=1.0)
    assert done


def test_empty_result_counts_failure_not_progress(tmp_path):
    state = make_state(tmp_path)
    items, _ = state.lease("phone", 1, now=0.0)
    reply = state.ingest("phone", items[0].id, "no json here")
    assert reply["accepted"] == 0
    assert state.snapshot()["empty_results"] == 1


def test_ingest_raw_parity_with_synth_records(tmp_path):
    """Mesh ingest must produce byte-identical records to local synth."""
    spec = prepare_synth("tools", "travel bot", None, _schemas_file(tmp_path))
    seen = set()
    records, parsed = ingest_raw(json.dumps(GOOD_TOOL), spec, True, seen)
    assert parsed == 1
    roles = [m["role"] for m in records[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert records[0]["messages"][0]["content"] == "travel bot"
    assert records[0]["tools"] == SCHEMAS
    # second pass is a pure duplicate
    records, parsed = ingest_raw(json.dumps(GOOD_TOOL), spec, True, seen)
    assert (records, parsed) == ([], 1)


def test_preference_ingest_passthrough(tmp_path):
    spec = prepare_synth("preference", "be terse", None)
    pair = {"prompt": "hi", "chosen": "yo", "rejected": "hello there, dear user"}
    records, parsed = ingest_raw(json.dumps(pair), spec, True, set())
    assert records == [pair]
    assert parsed == 1
