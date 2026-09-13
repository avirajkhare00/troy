"""Dataset loading, format auto-detection, and conversion to mlx-lm formats."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import DataConfig


def _read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON file must contain a list of records")
        return data
    if suffix == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(
        f"Unsupported file type: {path.suffix} (use .jsonl, .json, or .csv)"
    )


def detect_format(sample: Dict[str, Any]) -> str:
    keys = set(sample)
    if {"prompt", "chosen", "rejected"} <= keys:
        return "preference"
    if "messages" in keys:
        return "chat"
    if "conversations" in keys:
        return "sharegpt"
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if {"prompt", "completion"} <= keys:
        return "completions"
    if "text" in keys:
        return "text"
    raise ValueError(
        f"Could not detect data format from keys: {sorted(keys)}. "
        "Set `data.format` explicitly in troy.yaml."
    )


_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _to_messages(record: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """Normalize an SFT record to mlx-lm chat format ({"messages": [...]})."""
    if fmt == "chat":
        return {"messages": record["messages"]}
    if fmt == "sharegpt":
        messages = [
            {"role": _SHAREGPT_ROLES[m["from"].lower()], "content": m["value"]}
            for m in record["conversations"]
        ]
        return {"messages": messages}
    if fmt == "alpaca":
        user = record["instruction"]
        if record.get("input"):
            user = f"{user}\n\n{record['input']}"
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        messages += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": record["output"]},
        ]
        return {"messages": messages}
    if fmt in ("completions", "text"):
        return record  # already an mlx-lm native format
    raise ValueError(f"Unexpected format: {fmt}")


def _normalize_preference(record: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a DPO record to string prompt/chosen/rejected."""

    def text_of(v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):  # message-list style
            return "\n".join(m.get("content", "") for m in v)
        raise ValueError(f"Cannot interpret preference field: {v!r}")

    return {
        "prompt": text_of(record["prompt"]),
        "chosen": text_of(record["chosen"]),
        "rejected": text_of(record["rejected"]),
    }


def load_and_prepare(
    cfg: DataConfig, task: str, seed: int = 0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Load train/valid records normalized for the task.

    Returns (train_records, valid_records, detected_format).
    """
    train_path = Path(cfg.train).expanduser()
    records = _read_records(train_path)
    if not records:
        raise ValueError(f"No records found in {train_path}")

    fmt = cfg.format if cfg.format != "auto" else detect_format(records[0])

    if task in ("dpo", "orpo") and fmt != "preference":
        raise ValueError(
            f"Task `{task}` needs preference data (prompt/chosen/rejected); "
            f"detected format `{fmt}`."
        )
    if task == "sft" and fmt == "preference":
        raise ValueError(
            "Preference data detected — set `task: dpo` (or `task: orpo`) in troy.yaml."
        )

    if fmt == "preference":
        records = [_normalize_preference(r) for r in records]
    else:
        records = [_to_messages(r, fmt) for r in records]

    if cfg.valid:
        valid = _read_records(Path(cfg.valid).expanduser())
        valid = (
            [_normalize_preference(r) for r in valid]
            if fmt == "preference"
            else [_to_messages(r, fmt) for r in valid]
        )
        return records, valid, fmt

    # Split off validation
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = int(len(records) * cfg.val_split)
    val_idx = set(indices[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_idx]
    valid = [r for i, r in enumerate(records) if i in val_idx]
    return train, valid, fmt


# Required non-empty string fields per format (nested fields checked separately).
_REQUIRED_FIELDS = {
    "alpaca": ("instruction", "output"),
    "completions": ("prompt", "completion"),
    "preference": ("prompt", "chosen", "rejected"),
    "text": ("text",),
}


def _record_issues(record: Dict[str, Any], fmt: str, line: int) -> List[str]:
    issues = []

    def empty(v: Any) -> bool:
        return not (isinstance(v, str) and v.strip())

    for field in _REQUIRED_FIELDS.get(fmt, ()):
        if empty(record.get(field)):
            issues.append(f"line {line}: empty or missing `{field}`")
    if fmt == "chat":
        msgs = record.get("messages")
        if not isinstance(msgs, list) or not msgs:
            issues.append(f"line {line}: `messages` is not a non-empty list")
        else:
            roles = [m.get("role") for m in msgs if isinstance(m, dict)]
            if "assistant" not in roles:
                issues.append(f"line {line}: no assistant message")
            if any(
                not isinstance(m, dict) or empty(m.get("content"))
                for m in msgs
            ):
                issues.append(f"line {line}: message with empty content")
    if fmt == "sharegpt":
        convs = record.get("conversations")
        if not isinstance(convs, list) or not convs:
            issues.append(f"line {line}: `conversations` is not a non-empty list")
        else:
            bad = [
                m.get("from")
                for m in convs
                if not isinstance(m, dict)
                or str(m.get("from", "")).lower() not in _SHAREGPT_ROLES
            ]
            if bad:
                issues.append(f"line {line}: unknown speaker role(s): {bad}")
    if fmt == "preference" and not issues:
        if record["chosen"].strip() == record["rejected"].strip():
            issues.append(f"line {line}: chosen == rejected")
    return issues


def validate_records(path: str, max_issues: int = 50) -> Dict[str, Any]:
    """Lint a dataset: per-record issues, mixed formats, duplicates.

    Returns {"records", "format", "issues", "duplicates", "truncated"}.
    """
    p = Path(path).expanduser()
    issues: List[str] = []
    records: List[Tuple[int, Dict[str, Any]]] = []

    if p.suffix.lower() == ".jsonl":
        with open(p) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append((i, json.loads(line)))
                except json.JSONDecodeError as e:
                    issues.append(f"line {i}: invalid JSON ({e.msg})")
    else:
        records = [(i, r) for i, r in enumerate(_read_records(p), 1)]

    if not records:
        return {
            "records": 0, "format": "unknown", "issues": issues or ["file has no records"],
            "duplicates": 0, "truncated": False,
        }

    try:
        fmt = detect_format(records[0][1])
    except ValueError as e:
        return {
            "records": len(records), "format": "unknown",
            "issues": issues + [str(e)], "duplicates": 0, "truncated": False,
        }

    seen: Dict[str, int] = {}
    duplicates = 0
    for line, record in records:
        try:
            rec_fmt = detect_format(record)
        except ValueError:
            issues.append(f"line {line}: keys match no known format")
            continue
        if rec_fmt != fmt:
            issues.append(f"line {line}: format `{rec_fmt}` (file is `{fmt}`)")
            continue
        issues.extend(_record_issues(record, fmt, line))
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            duplicates += 1
            issues.append(f"line {line}: exact duplicate of line {seen[key]}")
        else:
            seen[key] = line

    truncated = len(issues) > max_issues
    return {
        "records": len(records),
        "format": fmt,
        "issues": issues[:max_issues],
        "duplicates": duplicates,
        "truncated": truncated,
    }


def inspect_stats(path: str) -> Dict[str, Any]:
    """Lightweight dataset statistics for `troy data inspect`-style output."""
    records = _read_records(Path(path).expanduser())
    fmt = detect_format(records[0]) if records else "unknown"
    lengths = [len(json.dumps(r)) for r in records]
    return {
        "records": len(records),
        "format": fmt,
        "avg_chars": sum(lengths) / max(len(lengths), 1),
        "max_chars": max(lengths, default=0),
    }
