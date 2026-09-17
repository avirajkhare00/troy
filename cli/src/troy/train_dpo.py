"""Direct Preference Optimization on Apple Silicon.

Memory trick: with LoRA, the frozen reference model is the policy model with
every adapter's scale set to 0 — so DPO needs no second copy of the weights,
which matters on unified memory.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load

from .config import TroyConfig
from .train_sft import apply_lora, resolve_batch_size, resolve_iters, save_adapter_config


def _lora_modules(model) -> list:
    return [
        m
        for _, m in model.named_modules()
        if hasattr(m, "lora_a") and hasattr(m, "scale")
    ]


class _ReferenceMode:
    """Context manager: zero every LoRA scale so the model acts as the frozen base."""

    def __init__(self, model):
        self.modules = _lora_modules(model)
        self.saved: List[float] = []

    def __enter__(self):
        self.saved = [m.scale for m in self.modules]
        for m in self.modules:
            m.scale = 0.0

    def __exit__(self, *exc):
        for m, s in zip(self.modules, self.saved):
            m.scale = s


def _encode_pair(tokenizer, prompt: str, completion: str, max_len: int) -> Tuple[List[int], int]:
    """Tokenize prompt+completion with the chat template; return (tokens, prompt_len)."""
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False
    )
    full = messages + [{"role": "assistant", "content": completion}]
    full_tokens = tokenizer.apply_chat_template(full, return_dict=False)
    full_tokens = full_tokens[:max_len]
    prompt_len = min(len(prompt_tokens), len(full_tokens) - 1)
    return full_tokens, prompt_len


def _batch(
    pairs: List[Tuple[List[int], int]], pad_id: int
) -> Tuple[mx.array, mx.array]:
    """Pad a list of (tokens, prompt_len) into (tokens, completion_mask) arrays."""
    max_len = max(len(t) for t, _ in pairs)
    tokens, mask = [], []
    for t, plen in pairs:
        pad = max_len - len(t)
        tokens.append(list(t) + [pad_id] * pad)
        # mask over positions whose *target* token is part of the completion
        m = [1.0 if plen <= j < len(t) else 0.0 for j in range(1, max_len)]
        mask.append(m)
    return mx.array(tokens), mx.array(mask)


def _sequence_logps(model, tokens: mx.array, mask: mx.array) -> mx.array:
    """Sum of per-token log-probs over the completion for each sequence."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    logps = -nn.losses.cross_entropy(logits, targets, reduction="none")
    return (logps * mask).sum(axis=-1)


def train_pairs(
    config: TroyConfig,
    train_records: List[Dict[str, Any]],
    task: str,
    make_loss: Callable[[Any, Any], Callable],
    hyper: str,
) -> None:
    """Shared LoRA loop for pairwise preference tasks (DPO, ORPO).

    make_loss(model, tokenizer) returns loss_fn(model, tc, mc, tr, mr) ->
    (loss, reward_acc); a task may wrap it with a per-batch prologue (see
    DPO's reference pass) by returning a callable that closes over the batch.
    """
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

    max_len = config.training.seq_len
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    encoded = [
        (
            _encode_pair(tokenizer, r["prompt"], r["chosen"], max_len),
            _encode_pair(tokenizer, r["prompt"], r["rejected"], max_len),
        )
        for r in train_records
    ]

    step = make_loss(model, tokenizer)
    opt = optim.Adam(learning_rate=config.training.lr)
    print(
        f"Training: task={task} {hyper} batch_size={batch_size} iters={iters} "
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

        (loss, acc), grads = step(model, tc, mc, tr, mr)
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


def run_dpo(config: TroyConfig, train_records: List[Dict[str, Any]]) -> None:
    beta = config.training.dpo.beta

    def make_loss(model, tokenizer):
        def loss_fn(model, tc, mc, tr, mr, ref_c, ref_r):
            pol_c = _sequence_logps(model, tc, mc)
            pol_r = _sequence_logps(model, tr, mr)
            logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
            return -nn.log_sigmoid(logits).mean(), (logits > 0).mean()

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step(model, tc, mc, tr, mr):
            with _ReferenceMode(model):
                ref_c = mx.stop_gradient(_sequence_logps(model, tc, mc))
                ref_r = mx.stop_gradient(_sequence_logps(model, tr, mr))
                mx.eval(ref_c, ref_r)
            return loss_and_grad(model, tc, mc, tr, mr, ref_c, ref_r)

        return step

    train_pairs(config, train_records, "dpo", make_loss, f"beta={beta}")
