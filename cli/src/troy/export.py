"""Merge adapters and export to deployment formats (MLX, GGUF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_export(
    base: str,
    adapter_path: str,
    save_path: str,
    fmt: str = "mlx",
    dequantize: bool = False,
) -> None:
    """Fuse a LoRA adapter into the base model via mlx-lm.

    fmt: "mlx" (fused MLX weights) or "gguf" (also writes ggml-model-f16.gguf
    for llama.cpp / Ollama / LM Studio; supported for Llama, Mistral and
    Mixtral architectures by mlx-lm).
    """
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        base,
        "--adapter-path",
        adapter_path,
        "--save-path",
        save_path,
    ]
    if dequantize:
        cmd.append("--dequantize")
    if fmt == "gguf":
        cmd.append("--export-gguf")
    print("Fusing adapter into base model ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"Export complete: {Path(save_path).resolve()}")
    if fmt == "gguf":
        print("GGUF file written alongside the fused model (ggml-model-f16.gguf).")
