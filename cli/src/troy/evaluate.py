"""Evaluate a trained adapter: base vs tuned perplexity and side-by-side samples.

The base-model numbers reuse the zero-scale trick: with every LoRA scale set
to 0 the tuned model *is* the base model, so both sides come from one load.
"""

from __future__ import annotations

import math
import types
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import CacheDataset, create_dataset
from mlx_lm.tuner.trainer import default_loss, evaluate, iterate_batches
from mlx_lm.utils import load

from .config import TroyConfig
from .train_dpo import _ReferenceMode


def _val_loss(model, dataset, batch_size: int, seq_len: int) -> float:
    return float(
        evaluate(
            model=model,
            dataset=CacheDataset(dataset),
            batch_size=batch_size,
            num_batches=-1,
            max_seq_length=seq_len,
            loss=default_loss,
            iterate_batches=iterate_batches,
        )
    )


def run_eval(
    config: TroyConfig,
    valid_records: List[Dict[str, Any]],
    sample_prompts: Optional[List[str]] = None,
    max_tokens: int = 200,
) -> None:
    adapter_file = config.adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        raise SystemExit(f"No adapter at {adapter_file}. Run `troy train` first.")

    print(f"Loading {config.base} + adapter ...")
    model, tokenizer = load(config.base, adapter_path=str(config.adapter_path))

    results = {}
    if valid_records and config.task == "sft":
        ds_config = types.SimpleNamespace(mask_prompt=config.data.mask_prompt)
        dataset = create_dataset(valid_records, tokenizer, ds_config)
        bs = max(1, min(4, len(valid_records)))

        tuned = _val_loss(model, dataset, bs, config.training.seq_len)
        with _ReferenceMode(model):
            base = _val_loss(model, dataset, bs, config.training.seq_len)
        results = {
            "val records": len(valid_records),
            "base loss": f"{base:.3f}",
            "tuned loss": f"{tuned:.3f}",
            "base ppl": f"{math.exp(base):.1f}",
            "tuned ppl": f"{math.exp(tuned):.1f}",
        }
        for k, v in results.items():
            print(f"  {k:>12}: {v}")
        delta = base - tuned
        print(f"  {'Δ loss':>12}: {delta:+.3f} ({'tuned better' if delta > 0 else 'base better'})")
    elif config.task != "sft":
        print("(loss comparison is defined for task: sft — showing generations only)")

    if sample_prompts:
        sampler = make_sampler(temp=0.0)
        print("\n--- side-by-side generations (temperature 0) ---")
        for prompt in sample_prompts:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=False,
            )
            tuned_out = generate(model, tokenizer, templated, max_tokens=max_tokens, sampler=sampler)
            with _ReferenceMode(model):
                base_out = generate(model, tokenizer, templated, max_tokens=max_tokens, sampler=sampler)
            print(f"\n>> {prompt}")
            print(f"[base]  {base_out.strip()[:400]}")
            print(f"[tuned] {tuned_out.strip()[:400]}")
