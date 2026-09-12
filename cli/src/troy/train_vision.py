"""Vision-language fine-tuning (wraps mlx-vlm's LoRA trainer).

Contract: `data.train` is a FOLDER containing images plus a metadata.jsonl
with {"file_name", "question", "answer"} rows (HF imagefolder format).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from .config import TroyConfig


def ensure_mlx_vlm() -> None:
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Vision fine-tuning needs mlx-vlm. Install with:\n"
            "  pip install 'troy-cli[vision]'"
        )


def count_records(data_dir: Path) -> int:
    meta = data_dir / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(
            f"{data_dir} has no metadata.jsonl. Vision data is a folder of images "
            'plus metadata.jsonl rows: {"file_name": ..., "question": ..., "answer": ...}'
        )
    with open(meta) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if rows and not {"file_name", "question", "answer"} <= set(rows[0]):
        raise SystemExit(
            "metadata.jsonl rows need file_name, question, and answer fields."
        )
    return len(rows)


def run_vision_sft(config: TroyConfig) -> None:
    ensure_mlx_vlm()
    data_dir = Path(config.data.train).expanduser()
    n = count_records(data_dir)
    print(f"Vision data: {n} images (folder: {data_dir})")

    t = config.training
    batch_size = 1 if t.batch_size == "auto" else int(t.batch_size)
    iters = t.iters or max(1, math.ceil((t.epochs or 3) * n / batch_size))

    config.adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", config.base,
        "--dataset", str(data_dir),
        "--split", "train",
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(t.lr),
        "--lora-rank", str(t.lora.r),
        "--lora-alpha", str(t.lora.alpha),
        "--lora-dropout", str(t.lora.dropout),
        "--max-seq-length", str(t.seq_len),
        "--steps-per-save", str(t.save_every),
        "--output-path", str(config.adapter_path),
    ]
    if t.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    print(f"Training: task=sft (vision) batch_size={batch_size} iters={iters} lr={t.lr}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"\nDone. Adapter saved to {config.adapter_path}")


def run_vision_chat(
    model: str, adapter: str | None, image: str, prompt: str, max_tokens: int, temperature: float
) -> None:
    ensure_mlx_vlm()
    cmd = [
        sys.executable, "-m", "mlx_vlm", "generate",
        "--model", model,
        "--image", image,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]
    if adapter:
        cmd += ["--adapter-path", adapter]
    raise SystemExit(subprocess.run(cmd).returncode)
