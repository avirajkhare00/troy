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


# model_type values registered in mlx-swift-lm's LLMTypeRegistry (MLXLLM).
# Fused models outside this set won't load in an iOS app using mlx-swift-lm.
_IOS_SUPPORTED_MODEL_TYPES = {
    "mistral", "mixtral", "llama", "phi", "phi3", "phimoe",
    "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n",
    "gemma4", "gemma4_unified", "gemma4_text",
    "qwen2", "qwen3", "qwen3_moe", "qwen3_next",
    "qwen3_5", "qwen3_5_moe", "qwen3_5_text",
    "minicpm", "starcoder2", "cohere", "openelm", "internlm2",
    "deepseek_v2", "deepseek_v3", "granite", "helium", "granitemoehybrid",
    "glm4", "glm4_moe", "glm4_moe_lite", "falcon_h1", "bitnet", "smollm3",
    "ernie4_5", "lfm2", "lfm2_moe", "exaone4", "gpt_oss", "olmoe", "olmo2",
    "olmo3", "nemotron_h", "jamba", "mamba2", "mistral3", "apertus",
}

# On an 8 GB iPhone, an app with the Increased Memory Limit entitlement gets
# roughly 6 GB resident; weights + KV cache + app overhead must fit inside it.
_IOS_COMFORTABLE_GB = 2.2   # loads without entitlements on recent iPhones
_IOS_MAX_GB = 4.0           # needs Increased Memory Limit / Extended Virtual Addressing

_IOS_README = """\
# Run this model on iPhone / iPad with MLX Swift

This directory is a fused MLX model in the layout `mlx-swift-lm` loads directly.

## Load it

Add the Swift package: https://github.com/ml-explore/mlx-swift-lm

```swift
import MLXLLM
import MLXLMCommon

// From a local directory bundled with (or downloaded by) your app:
let modelDirectory: URL = ...  // this folder on device
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(directory: modelDirectory))

let result = try await container.perform { context in
    let input = try await context.processor.prepare(
        input: .init(prompt: "Hello!"))
    return try MLXLMCommon.generate(
        input: input, parameters: .init(), context: context)
}
```

Weights this large should not ship inside the app bundle — download them on
first launch (Background Assets, or a direct download from your server or the
Hugging Face Hub via `troy push`).

## Memory entitlements

Models over ~2 GB need one of these capabilities in Xcode
(Signing & Capabilities → + Capability):

- **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`)
- **Extended Virtual Addressing**

Either is usually sufficient; devices with 8 GB RAM run 4-bit models up to
roughly 4 GB of weights. Test on the oldest device you target.
"""


def _base_is_quantized(base: str) -> bool:
    import json

    p = Path(base)
    if not p.exists():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(base, allow_patterns=["config.json"]))
    return "quantization" in json.loads((p / "config.json").read_text())


def _fuse_for_ios(base: str, adapter_path: str, save_path: str) -> None:
    """Fuse for iOS without destroying the adapter.

    Fusing a LoRA into 4-bit weights loses the deltas to quantization noise —
    the exported model silently behaves like the base. So for a quantized
    base: dequantize-fuse to fp16, then requantize fresh at 8 bits (verified
    to preserve tuned behavior; 6 bits already degrades it).
    """
    import shutil
    import tempfile

    if not _base_is_quantized(base):
        _fuse(base, adapter_path, save_path, dequantize=False)
        return
    from mlx_lm import convert

    print("Quantized base: dequantize-fusing, then requantizing at 8 bits")
    print("(fusing straight into 4-bit silently erases the adapter).")
    out = Path(save_path)
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as td:
        _fuse(base, adapter_path, td, dequantize=True)
        convert(td, mlx_path=str(out), quantize=True, q_bits=8)


def _dir_weight_bytes(save_path: Path) -> int:
    return sum(p.stat().st_size for p in save_path.glob("*.safetensors"))


def _export_ios(save_path: Path) -> None:
    """Validate a fused MLX directory for mlx-swift-lm on iOS and add a how-to."""
    import json

    with open(save_path / "config.json") as f:
        config = json.load(f)

    problems = []
    model_type = config.get("model_type", "?")
    if model_type not in _IOS_SUPPORTED_MODEL_TYPES:
        problems.append(
            f"model_type `{model_type}` is not in mlx-swift-lm's registry — "
            "it won't load on iOS without adding the architecture in Swift."
        )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (save_path / name).exists():
            problems.append(
                f"{name} is missing — swift-transformers needs it to build "
                "the tokenizer on device."
            )

    gb = _dir_weight_bytes(save_path) / 1e9
    quantized = "quantization" in config
    print(f"Weights: {gb:.2f} GB ({'quantized' if quantized else 'float'}), "
          f"model_type: {model_type}")
    if not quantized:
        print("Tip: unquantized weights are heavy for iPhone — train from a "
              "4-bit base (e.g. an mlx-community *-4bit model) for iOS.")
    if gb <= _IOS_COMFORTABLE_GB:
        print("iPhone fit: comfortable — loads on recent iPhones; the "
              "Increased Memory Limit capability is still recommended.")
    elif gb <= _IOS_MAX_GB:
        print("iPhone fit: needs the Increased Memory Limit or Extended "
              "Virtual Addressing capability; expect 8 GB-RAM devices only.")
    else:
        print(f"iPhone fit: unlikely — {gb:.1f} GB of weights exceeds what "
              "an 8 GB iPhone can keep resident. Use a smaller or more "
              "aggressively quantized base.")

    (save_path / "README-iOS.md").write_text(_IOS_README)
    print(f"Wrote {save_path / 'README-iOS.md'}")

    if problems:
        for p in problems:
            print(f"WARNING: {p}")
        raise SystemExit(1)


def run_export(
    base: str,
    adapter_path: str,
    save_path: str,
    fmt: str = "mlx",
    dequantize: bool = False,
) -> None:
    """Fuse a LoRA adapter into the base model; optionally convert to GGUF.

    fmt: "mlx" (fused MLX weights), "gguf" (also writes ggml-model-f16.gguf
    for llama.cpp / Ollama / LM Studio; llama, mistral and mixtral
    architectures, unquantized base), or "ios" (fused MLX weights validated
    for mlx-swift-lm on iPhone/iPad, plus a README-iOS.md how-to).
    """
    if fmt == "ios" and dequantize:
        print("Note: --dequantize with -f ios makes weights ~4x larger; "
              "quantized weights are what you want on iPhone.")
    if not Path(base).exists():
        # mlx-lm's training path downloads a partial snapshot; fuse checks for
        # a complete one. Top it up before fusing.
        from huggingface_hub import snapshot_download

        snapshot_download(base)

    if fmt == "ios":
        _fuse_for_ios(base, adapter_path, save_path)
    else:
        _fuse(base, adapter_path, save_path, dequantize)
    print(f"Fused model: {Path(save_path).resolve()}")
    if fmt == "gguf":
        out = _export_gguf(Path(save_path))
        print(f"GGUF: {out.resolve()}")
    elif fmt == "ios":
        _export_ios(Path(save_path))
