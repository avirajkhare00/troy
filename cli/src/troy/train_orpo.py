"""ORPO: Odds Ratio Preference Optimization on Apple Silicon.

ORPO needs no reference model at all — the loss combines a standard SFT term
on the chosen response with an odds-ratio penalty pushing chosen above
rejected. One model in memory, one pass per completion.
"""

from __future__ import annotations

from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn

from .config import TroyConfig
from .train_dpo import train_pairs


def _mean_logps(model, tokens: mx.array, mask: mx.array):
    """Per-sequence mean log-prob over completion tokens (and the sum + count)."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    logps = -nn.losses.cross_entropy(logits, targets, reduction="none")
    total = (logps * mask).sum(axis=-1)
    count = mx.maximum(mask.sum(axis=-1), 1)
    return total / count, total, count


def run_orpo(config: TroyConfig, train_records: List[Dict[str, Any]]) -> None:
    lam = config.training.orpo.lam

    def make_loss(model, tokenizer):
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
            return sft_loss + lam * or_loss, (mean_c > mean_r).mean()

        return nn.value_and_grad(model, loss_fn)

    train_pairs(config, train_records, "orpo", make_loss, f"lambda={lam}")
