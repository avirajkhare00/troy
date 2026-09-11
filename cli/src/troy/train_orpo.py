"""ORPO: Odds Ratio Preference Optimization on Apple Silicon.

ORPO needs no reference model at all — the loss combines a standard SFT term
on the chosen response with an odds-ratio penalty pushing chosen above
rejected. One model in memory, one pass per completion.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load

from .config import TroyConfig
from .train_dpo import _batch, _encode_pair
from .train_sft import apply_lora, resolve_batch_size, resolve_iters, save_adapter_config


def _mean_logps(model, tokens: mx.array, mask: mx.array):
    """Per-sequence mean log-prob over completion tokens (and the sum + count)."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    logps = -nn.losses.cross_entropy(logits, targets, reduction="none")
    total = (logps * mask).sum(axis=-1)
    count = mx.maximum(mask.sum(axis=-1), 1)
    return total / count, total, count


def run_orpo(
    config: TroyConfig,
    train_records: List[Dict[str, Any]],
    valid_records: List[Dict[str, Any]],
) -> None:
    mx.random.seed(config.training.seed)
    print(f"Loading {config.base} ...")
    model, tokenizer = load(config.base)

    batch_size = resolve_batch_size(config)
    if config.training.batch_size == "auto":
        batch_size = max(1, batch_size // 2)  # chosen + rejected per example
    iters = resolve_iters(config, len(train_records), batch_size)
    num_layers = apply_lora(model, config)
    save_adapter_config(config, num_layers)
    if config.training.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    lam = config.training.orpo.lam
    max_len = config.training.seq_len
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    encoded = [
        (
            _encode_pair(tokenizer, r["prompt"], r["chosen"], max_len),
            _encode_pair(tokenizer, r["prompt"], r["rejected"], max_len),
        )
        for r in train_records
    ]

    def loss_fn(model, tc, mc, tr, mr):
        mean_c, total_c, count_c = _mean_logps(model, tc, mc)
        mean_r, _, _ = _mean_logps(model, tr, mr)
        # log odds: log(p/(1-p)) with p = exp(mean logp)
        log_odds = (mean_c - mean_r) - (
            mx.log1p(-mx.exp(mx.minimum(mean_c, -1e-6)))
            - mx.log1p(-mx.exp(mx.minimum(mean_r, -1e-6)))
        )
        or_loss = -nn.log_sigmoid(log_odds).mean()
        sft_loss = -(total_c.sum() / count_c.sum())
        reward_acc = (mean_c > mean_r).mean()
        return sft_loss + lam * or_loss, reward_acc

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=config.training.lr)

    print(
        f"Training: task=orpo lambda={lam} batch_size={batch_size} iters={iters} "
        f"lr={config.training.lr} pairs={len(encoded)}"
    )

    config.adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = config.adapter_path / "adapters.safetensors"
    n = len(encoded)
    losses, accs = [], []
    start = time.time()

    for it in range(iters):
        idx = [(it * batch_size + k) % n for k in range(batch_size)]
        tc, mc = _batch([encoded[i][0] for i in idx], pad_id)
        tr, mr = _batch([encoded[i][1] for i in idx], pad_id)

        (loss, acc), grads = loss_and_grad(model, tc, mc, tr, mr)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(loss.item())
        accs.append(acc.item())

        if (it + 1) % 10 == 0 or it == iters - 1:
            speed = (it + 1) / (time.time() - start)
            print(
                f"Iter {it + 1}/{iters}: loss {sum(losses)/len(losses):.4f}, "
                f"reward acc {sum(accs)/len(accs):.3f}, {speed:.2f} it/s"
            )
            losses, accs = [], []

        if (it + 1) % config.training.save_every == 0:
            _save(model, adapter_file)

    _save(model, adapter_file)
    print(f"\nDone. Adapter saved to {config.adapter_path}")


def _save(model, adapter_file) -> None:
    mx.save_safetensors(str(adapter_file), dict(tree_flatten(model.trainable_parameters())))
