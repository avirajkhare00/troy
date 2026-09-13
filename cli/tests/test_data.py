import json

import pytest

from troy.config import DataConfig
from troy.data import detect_format, load_and_prepare


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_detect_formats():
    assert detect_format({"instruction": "a", "output": "b"}) == "alpaca"
    assert detect_format({"conversations": []}) == "sharegpt"
    assert detect_format({"messages": []}) == "chat"
    assert detect_format({"prompt": "a", "completion": "b"}) == "completions"
    assert detect_format({"prompt": "a", "chosen": "b", "rejected": "c"}) == "preference"
    assert detect_format({"text": "a"}) == "text"
    with pytest.raises(ValueError, match="detect"):
        detect_format({"foo": 1})


def test_alpaca_to_messages(tmp_path):
    f = tmp_path / "t.jsonl"
    write_jsonl(f, [{"instruction": "Q", "input": "extra", "output": "A"}] * 10)
    train, valid, fmt = load_and_prepare(DataConfig(train=str(f), val_split=0.2), "sft")
    assert fmt == "alpaca"
    assert len(train) == 8 and len(valid) == 2
    msgs = train[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "Q\n\nextra"}
    assert msgs[1] == {"role": "assistant", "content": "A"}


def test_sharegpt_role_mapping(tmp_path):
    f = tmp_path / "t.jsonl"
    write_jsonl(f, [{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}] * 5)
    train, _, fmt = load_and_prepare(DataConfig(train=str(f), val_split=0.0), "sft")
    assert fmt == "sharegpt"
    assert train[0]["messages"][0]["role"] == "user"
    assert train[0]["messages"][1]["role"] == "assistant"


def test_preference_normalizes_message_lists(tmp_path):
    f = tmp_path / "t.jsonl"
    write_jsonl(f, [{"prompt": "p", "chosen": [{"role": "assistant", "content": "good"}], "rejected": "bad"}] * 4)
    train, _, fmt = load_and_prepare(DataConfig(train=str(f), val_split=0.0), "dpo")
    assert fmt == "preference"
    assert train[0]["chosen"] == "good"


def test_task_format_mismatch(tmp_path):
    f = tmp_path / "t.jsonl"
    write_jsonl(f, [{"instruction": "Q", "output": "A"}])
    with pytest.raises(ValueError, match="preference"):
        load_and_prepare(DataConfig(train=str(f)), "dpo")
    write_jsonl(f, [{"prompt": "p", "chosen": "c", "rejected": "r"}])
    with pytest.raises(ValueError, match="task: dpo"):
        load_and_prepare(DataConfig(train=str(f)), "sft")


def test_csv_input(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("instruction,output\nQ1,A1\nQ2,A2\n")
    train, _, fmt = load_and_prepare(DataConfig(train=str(f), val_split=0.0), "sft")
    assert fmt == "alpaca" and len(train) == 2


def test_explicit_valid_file(tmp_path):
    t, v = tmp_path / "t.jsonl", tmp_path / "v.jsonl"
    write_jsonl(t, [{"text": "a"}] * 6)
    write_jsonl(v, [{"text": "b"}] * 2)
    train, valid, _ = load_and_prepare(DataConfig(train=str(t), valid=str(v)), "sft")
    assert len(train) == 6 and len(valid) == 2


def test_validate_clean_file(tmp_path):
    from troy.data import validate_records

    f = tmp_path / "t.jsonl"
    write_jsonl(f, [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5)])
    report = validate_records(str(f))
    assert report["format"] == "alpaca"
    assert report["records"] == 5
    assert report["issues"] == []


def test_validate_catches_problems(tmp_path):
    from troy.data import validate_records

    f = tmp_path / "t.jsonl"
    rows = [
        {"instruction": "q", "output": "a"},          # ok
        {"instruction": "", "output": "a"},           # empty field
        {"prompt": "p", "completion": "c"},           # mixed format
        {"instruction": "q", "output": "a"},          # duplicate of line 1
    ]
    with open(f, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write("{broken json\n")

    report = validate_records(str(f))
    text = "\n".join(report["issues"])
    assert "empty or missing `instruction`" in text
    assert "format `completions`" in text
    assert "duplicate of line 1" in text
    assert "invalid JSON" in text
    assert report["duplicates"] == 1


def test_validate_preference_and_chat(tmp_path):
    from troy.data import validate_records

    f = tmp_path / "p.jsonl"
    write_jsonl(f, [{"prompt": "p", "chosen": "same", "rejected": "same"}])
    assert "chosen == rejected" in validate_records(str(f))["issues"][0]

    f2 = tmp_path / "c.jsonl"
    write_jsonl(f2, [{"messages": [{"role": "user", "content": "hi"}]}])
    assert "no assistant message" in validate_records(str(f2))["issues"][0]
