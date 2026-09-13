import pytest

from troy.synth import chunk_text, collect_sources, parse_pairs, pick_teacher


def test_pick_teacher_scales_with_memory():
    assert "1.7B" in pick_teacher(8)
    assert "4B" in pick_teacher(16)
    assert "8B" in pick_teacher(24)
    assert "14B" in pick_teacher(36)
    assert "30B" in pick_teacher(64)


def test_chunk_text_short_passthrough():
    assert chunk_text("hello", size=100) == ["hello"]


def test_chunk_text_splits_on_paragraphs():
    text = ("para one " * 50 + "\n\n" + "para two " * 50).strip()
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)
    # nothing lost beyond whitespace at the seams
    assert "para two" in chunks[-1]


def test_parse_pairs_jsonl():
    raw = (
        '{"user": "What is Troy?", "assistant": "A fine-tuning CLI."}\n'
        '{"user": "Which Macs?", "assistant": "Apple Silicon, M1+."}\n'
    )
    pairs = parse_pairs(raw, ("user", "assistant"))
    assert len(pairs) == 2
    assert pairs[0]["user"] == "What is Troy?"


def test_parse_pairs_tolerates_noise():
    raw = (
        "<think>let me write some examples</think>\n"
        "Here are the examples:\n"
        "```json\n"
        '{"user": "q1",\n "assistant": "a1"}\n'
        "```\n"
        "not json at all {broken\n"
        '{"user": "q2", "assistant": "a2", "extra": 1}\n'
        '{"user": "", "assistant": "empty user skipped"}\n'
        '{"user": "no assistant key"}\n'
    )
    pairs = parse_pairs(raw, ("user", "assistant"))
    assert [p["user"] for p in pairs] == ["q1", "q2"]
    assert "extra" not in pairs[1]


def test_parse_pairs_preference_keys():
    raw = '{"prompt": "p", "chosen": "good", "rejected": "bad"}'
    pairs = parse_pairs(raw, ("prompt", "chosen", "rejected"))
    assert pairs == [{"prompt": "p", "chosen": "good", "rejected": "bad"}]


def test_collect_sources_file_and_folder(tmp_path):
    (tmp_path / "a.md").write_text("# Doc A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Doc B")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.md").write_text("hidden")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")

    sources = collect_sources(tmp_path)
    names = [n for n, _ in sources]
    assert names == ["a.md", "sub/b.txt"]

    single = collect_sources(tmp_path / "a.md")
    assert single == [("a.md", "# Doc A")]


def test_collect_sources_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        collect_sources(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No readable"):
        collect_sources(empty)
