"""Supervised fine-tuning via the mlx-lm tuner."""

from __future__ import annotations

import json
import math
import types
from typing import Any, Dict, List

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm.tuner.datasets import CacheDataset, create_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load

from .config import TroyConfig
from .hardware import auto_batch_size, detect


def resolve_batch_size(config: TroyConfig) -> int:
    if config.training.batch_size == "auto":
        return auto_batch_size(detect().memory_gb, config.training.seq_len)
    return int(config.training.batch_size)


def resolve_iters(config: TroyConfig, n_train: int, batch_size: int) -> int:
    t = config.training
    if t.iters:
        return t.iters
    epochs = t.epochs or 3
    return max(1, math.ceil(epochs * n_train / batch_size))


def apply_lora(model, config: TroyConfig):
    model.freeze()
    lora = config.training.lora
    n_model_layers = len(model.layers)
    num_layers = (
        n_model_layers if lora.layers == "all" else min(int(lora.layers), n_model_layers)
    )
    linear_to_lora_layers(
        model,
        num_layers,
        {"rank": lora.r, "scale": lora.scale, "dropout": lora.dropout},
    )
    print_trainable_parameters(model)
    return num_layers


def save_adapter_config(config: TroyConfig, num_layers: int) -> None:
    """Write the adapter_config.json mlx-lm needs to re-load the adapter."""
    lora = config.training.lora
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": lora.r,
            "scale": lora.scale,
            "dropout": lora.dropout,
        },
    }
    with open(config.adapter_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)


def run_sft(
    config: TroyConfig,
    train_records: List[Dict[str, Any]],
    valid_records: List[Dict[str, Any]],
) -> None:
    mx.random.seed(config.training.seed)
    print(f"Loading {config.base} ...")
    model, tokenizer = load(config.base)

    if not valid_records:  # trainer always evaluates; give it something tiny
        valid_records = train_records[:1]
    ds_config = types.SimpleNamespace(mask_prompt=config.data.mask_prompt)
    train_set = create_dataset(train_records, tokenizer, ds_config)
    valid_set = create_dataset(valid_records, tokenizer, ds_config)

    batch_size = resolve_batch_size(config)
    batch_size = max(1, min(batch_size, len(train_records), len(valid_records)))
    iters = resolve_iters(config, len(train_records), batch_size)
    num_layers = apply_lora(model, config)
    save_adapter_config(config, num_layers)

    args = TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        max_seq_length=config.training.seq_len,
        adapter_file=str(config.adapter_path / "adapters.safetensors"),
        grad_checkpoint=config.training.grad_checkpoint,
        grad_accumulation_steps=config.training.grad_accumulation_steps,
        steps_per_save=config.training.save_every,
        steps_per_eval=min(200, max(1, iters // 2)),
        val_batches=min(25, max(1, len(valid_records) // batch_size)),
    )
    opt = optim.Adam(learning_rate=config.training.lr)

    print(
        f"Training: task=sft batch_size={batch_size} iters={iters} "
        f"lr={config.training.lr} seq_len={config.training.seq_len}"
    )
    train(
        model=model,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=args,
    )
    print(f"\nDone. Adapter saved to {config.adapter_path}")
