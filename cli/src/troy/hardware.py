"""Apple Silicon hardware detection and guidance."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


@dataclass
class Hardware:
    chip: str
    memory_gb: float
    macos: str
    arch: str
    free_disk_gb: float

    @property
    def is_apple_silicon(self) -> bool:
        return is_apple_silicon()


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect() -> Hardware:
    mem_bytes = int(_sysctl("hw.memsize") or 0)
    total, used, free = shutil.disk_usage("/")
    return Hardware(
        chip=_sysctl("machdep.cpu.brand_string") or "unknown",
        memory_gb=mem_bytes / 2**30,
        macos=platform.mac_ver()[0] or "unknown",
        arch=platform.machine(),
        free_disk_gb=free / 2**30,
    )


# (min unified memory GB, guidance) — QLoRA 4-bit sizing EXAMPLES, not a
# whitelist: any mlx-lm-supported architecture trains.
MODEL_GUIDANCE = [
    (128, "up to ~70B (4-bit QLoRA), e.g. Llama-3.3-70B, Qwen2.5-72B"),
    (64, "up to ~32B (4-bit QLoRA), e.g. Qwen2.5-32B, Gemma-2-27B"),
    (36, "up to ~14B (4-bit QLoRA), e.g. Qwen2.5-14B, Phi-4"),
    (24, "up to ~8B (4-bit QLoRA), e.g. Llama-3.1-8B, Qwen3-8B"),
    (16, "up to ~4B (4-bit QLoRA), e.g. Qwen3-4B, Phi-3.5-mini, Gemma-3-4B"),
    (8, "up to ~1.5B (4-bit QLoRA), e.g. Qwen3-0.6B, Llama-3.2-1B"),
]


def model_guidance(memory_gb: float) -> str:
    for min_gb, text in MODEL_GUIDANCE:
        if memory_gb >= min_gb:
            return text
    return "very small models only (<1B)"


def auto_batch_size(memory_gb: float, seq_len: int) -> int:
    """Conservative batch size from unified memory and sequence length."""
    budget = max(memory_gb - 8, 1)  # leave room for the OS + model weights
    per_seq = seq_len / 2048  # rough scaling
    bs = int(budget // (4 * per_seq))
    return max(1, min(bs, 8))
