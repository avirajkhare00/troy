# Benchmarks

Honest, reproducible numbers from real runs. Nothing here is projected or
extrapolated — each row is a measured session, and the configs are in
[`examples/`](../examples/).

**Test machine:** MacBook Pro, Apple M1 Pro, 16 GB unified memory ·
macOS 26 · mlx 0.32.2 · mlx-lm 0.31.3 · troy 0.2.0

## Training throughput (LoRA r=8)

| Base model | Task | Batch × seq | Throughput | Peak memory |
|---|---|---|---|---|
| Qwen3-0.6B-4bit | SFT | 2 × 512 | 90–508 tok/s | 0.79 GB |
| Qwen3-0.6B-4bit | SFT | 2 × 1024 | 720–746 tok/s | 0.97 GB |
| Qwen3-0.6B-4bit | DPO | 2 pairs | 1.3–2.0 it/s | — |
| Qwen3-0.6B-4bit | ORPO | 2 pairs | 3.2 it/s | — |
| SmolLM2-135M | SFT | 2 × 1024 | (fast; used for CI) | < 1 GB |
| Llama-3.2-1B-4bit | SFT | 2 × 1024 | verified, not profiled | — |

Notes:
- SFT throughput varies with prompt caching across iterations; ranges show
  first→steady-state.
- ORPO is ~2× DPO throughput here: no reference-model forward passes.
- DPO/ORPO hold **one** copy of the weights: Troy recovers the DPO reference
  model by zeroing LoRA scales instead of loading a second model.

## Did training work? (`troy eval`, json-extractor example)

| | Base | Tuned (4 epochs, 43 examples) |
|---|---|---|
| Val loss | 3.315 | **0.249** |
| Val perplexity | 27.5 | **1.3** |

## Reproduce

```bash
cd examples/json-extractor
troy train && troy eval
```

Have numbers from a different chip (M2/M3/M4, more memory, bigger models)?
PRs adding rows are very welcome — include the exact config and `troy doctor` output.
