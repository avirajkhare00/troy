"""Synthesize training data with a local teacher model.

`troy data synth` turns your documents (or a task description) into a
train.jsonl — the teacher runs locally via mlx-lm, so nothing leaves the Mac.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hardware import detect

# Source files worth mining for Q&A pairs.
SOURCE_SUFFIXES = {
    ".md", ".mdx", ".txt", ".rst", ".adoc",
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".swift",
    ".yaml", ".yml", ".toml", ".json", ".html",
}

# (min unified memory GB, teacher) — instruct models, 4-bit; the teacher only
# runs inference, so it can be larger than what the same Mac can train.
TEACHER_GUIDANCE = [
    (64, "mlx-community/Qwen3-30B-A3B-4bit"),
    (36, "mlx-community/Qwen3-14B-4bit"),
    (24, "mlx-community/Qwen3-8B-4bit"),
    (16, "mlx-community/Qwen3-4B-4bit"),
    (0, "mlx-community/Qwen3-1.7B-4bit"),
]

PAIRS_PER_CALL = 10


def pick_teacher(memory_gb: Optional[float] = None) -> str:
    if memory_gb is None:
        memory_gb = detect().memory_gb
    for min_gb, model in TEACHER_GUIDANCE:
        if memory_gb >= min_gb:
            return model
    return TEACHER_GUIDANCE[-1][1]


def collect_sources(path: Path) -> List[Tuple[str, str]]:
    """Read (name, text) from a file or a folder of text-ish files."""
    path = path.expanduser()
    if path.is_file():
        return [(path.name, path.read_text(errors="ignore"))]
    if not path.is_dir():
        raise ValueError(f"Source not found: {path}")
    sources = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in f.relative_to(path).parts):
            continue  # skip hidden dirs/files (.git, .venv, ...)
        text = f.read_text(errors="ignore").strip()
        if text:
            sources.append((str(f.relative_to(path)), text))
    if not sources:
        raise ValueError(
            f"No readable text files under {path} "
            f"(looked for {', '.join(sorted(SOURCE_SUFFIXES))})"
        )
    return sources


def chunk_text(text: str, size: int = 4000, overlap: int = 200) -> List[str]:
    """Split on paragraph boundaries where possible, hard-split otherwise."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_pairs(raw: str, keys: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Extract JSON objects with the given string keys from teacher output.

    Tolerates thinking blocks, code fences, prose between objects, and
    objects that span multiple lines.
    """
    raw = _THINK_RE.sub("", raw)
    pairs = []
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        brace = raw.find("{", pos)
        if brace == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(raw[brace:])
            pos = brace + consumed
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if not isinstance(obj, dict):
            continue
        values = {k: obj.get(k) for k in keys}
        if all(isinstance(v, str) and v.strip() for v in values.values()):
            pairs.append({k: v.strip() for k, v in values.items()})
    return pairs


def _chat_prompt(task: str, material: Optional[str], n: int) -> str:
    lines = [
        f"You are creating fine-tuning data for this task: {task}",
        "",
        f"Write {n} diverse training examples. Output ONLY JSON objects, "
        'one per line, each exactly: {"user": "...", "assistant": "..."}',
        "Vary phrasing, length, and difficulty. No numbering, no commentary.",
    ]
    if material:
        lines += [
            "",
            "Ground every example in this source material — questions a reader "
            "would ask about it, answered faithfully from it:",
            "---",
            material,
            "---",
        ]
    return "\n".join(lines)


def _preference_prompt(task: str, material: Optional[str], n: int) -> str:
    lines = [
        f"You are creating preference-tuning data for this task: {task}",
        "",
        f"Write {n} diverse training examples. Output ONLY JSON objects, one "
        'per line, each exactly: '
        '{"prompt": "...", "chosen": "...", "rejected": "..."}',
        "`chosen` is a genuinely good response; `rejected` is plausible but "
        "clearly worse for the task (vague, bloated, off-style, or subtly "
        "wrong). No numbering, no commentary.",
    ]
    if material:
        lines += [
            "",
            "Ground every example in this source material:",
            "---",
            material,
            "---",
        ]
    return "\n".join(lines)


def run_synth(
    n: int,
    out_path: Path,
    fmt: str,
    teacher: str,
    seed_task: Optional[str],
    source: Optional[Path],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Generate n examples; returns stats. Writes JSONL to out_path."""
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    chunks: List[Optional[str]]
    if source is not None:
        texts = collect_sources(source)
        chunks = [c for _, text in texts for c in chunk_text(text)]
    else:
        chunks = [None]  # pure seed-description mode

    task = seed_task or (
        "answering questions about the source material accurately and concisely"
    )
    keys = ("prompt", "chosen", "rejected") if fmt == "preference" else ("user", "assistant")
    build = _preference_prompt if fmt == "preference" else _chat_prompt

    print(f"Loading teacher {teacher} ...")
    model, tokenizer = load(teacher)
    sampler = make_sampler(temp=temperature)

    seen = set()
    records: List[Dict[str, Any]] = []
    calls = failures = 0
    chunk_i = 0
    max_calls = (n // PAIRS_PER_CALL + 1) * 4  # generous retry budget

    while len(records) < n and calls < max_calls:
        want = min(PAIRS_PER_CALL, n - len(records))
        prompt = build(task, chunks[chunk_i % len(chunks)], want)
        chunk_i += 1
        calls += 1
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )
        raw = generate(
            model, tokenizer, templated, max_tokens=max_tokens, sampler=sampler
        )
        pairs = parse_pairs(raw, keys)
        if not pairs:
            failures += 1
            continue
        for pair in pairs:
            key = pair[keys[0]].lower()
            if key in seen:
                continue
            seen.add(key)
            if fmt == "preference":
                records.append(pair)
            else:
                records.append(
                    {
                        "messages": [
                            {"role": "user", "content": pair["user"]},
                            {"role": "assistant", "content": pair["assistant"]},
                        ]
                    }
                )
            if len(records) >= n:
                break
        print(f"  {len(records)}/{n} examples ({calls} teacher calls)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "records": len(records),
        "requested": n,
        "teacher_calls": calls,
        "empty_responses": failures,
        "chunks": 0 if chunks == [None] else len(chunks),
        "out": str(out_path),
    }
