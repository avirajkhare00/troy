"""Merge adapters and export to deployment formats (MLX, GGUF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _fuse(base: str, adapter_path: str, save_path: str, dequantize: bool) -> None:
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
    print("Fusing adapter into base model ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _export_gguf(save_path: Path) -> Path:
    """Convert a fused MLX model directory to GGUF (llama/mistral/mixtral archs)."""
    import json

    import mlx.core as mx
    from mlx_lm import gguf as gguf_mod

    with open(save_path / "config.json") as f:
        config = json.load(f)

    weights = {}
    for part in sorted(save_path.glob("*.safetensors")):
        weights.update(mx.load(str(part)))

    # mlx-lm's converter permutes attention weights into non-contiguous views,
    # which save_gguf rejects — force contiguity on its output.
    orig_permute = gguf_mod.permute_weights
    gguf_mod.permute_weights = lambda *a, **k: mx.contiguous(orig_permute(*a, **k))
    try:
        out = save_path / "ggml-model-f16.gguf"
        gguf_mod.convert_to_gguf(save_path, weights, config, str(out))
    finally:
        gguf_mod.permute_weights = orig_permute
    return out


def run_export(
    base: str,
    adapter_path: str,
    save_path: str,
    fmt: str = "mlx",
    dequantize: bool = False,
) -> None:
    """Fuse a LoRA adapter into the base model; optionally convert to GGUF.

    fmt: "mlx" (fused MLX weights) or "gguf" (also writes ggml-model-f16.gguf
    for llama.cpp / Ollama / LM Studio; llama, mistral and mixtral
    architectures, unquantized base).
    """
    if not Path(base).exists():
        # mlx-lm's training path downloads a partial snapshot; fuse checks for
        # a complete one. Top it up before fusing.
        from huggingface_hub import snapshot_download

        snapshot_download(base)

    _fuse(base, adapter_path, save_path, dequantize)
    print(f"Fused model: {Path(save_path).resolve()}")
    if fmt == "gguf":
        out = _export_gguf(Path(save_path))
        print(f"GGUF: {out.resolve()}")
