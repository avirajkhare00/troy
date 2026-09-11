# Troy

**Fine-tune LLMs on your MacBook with one YAML file.**

Troy is a command-line tool for fine-tuning and preference-tuning language
models locally on Apple Silicon. No CUDA, no cloud, no training pipeline —
write a config, run one command, and train on the machine you already own.

Built on [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), Apple's ML framework for
Apple Silicon. Unified memory means a 36 GB MacBook fine-tunes models that
need a workstation GPU anywhere else.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.10–3.12

## Install

```bash
brew tap avirajkhare00/troy https://github.com/avirajkhare00/troy-homebrew
brew install troy
# or from source: pip install ./cli
```

## Quickstart

```bash
troy doctor            # check your Mac: chip, memory, MLX, what you can train
troy init              # create troy.yaml + sample data
troy train             # fine-tune (LoRA/QLoRA via MLX)
troy chat              # talk to the result
troy serve             # OpenAI-compatible API at localhost:8080/v1
troy export -f gguf    # ship it to llama.cpp / Ollama / LM Studio
```

## The config is the interface

```yaml
base: mlx-community/Qwen3-0.6B-4bit
task: sft            # or: dpo

data:
  train: ./data/train.jsonl   # alpaca, sharegpt, chat, completions, text — auto-detected
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto   # sized from your Mac's unified memory
  lora:
    r: 8
    alpha: 16

output: ./output
```

## Commands

| Command | Purpose |
|---|---|
| `troy init` | Create a config from a template (`chat`, `dpo`) |
| `troy doctor` | Hardware + dependency check, with model-size guidance |
| `troy train` | LoRA fine-tuning: SFT or DPO |
| `troy chat` | Interactive REPL (or `-p` for one-shot) with your adapter |
| `troy serve` | OpenAI-compatible API server for your model |
| `troy export` | Fuse the adapter; export MLX or GGUF |
| `troy data inspect` | Dataset stats and format detection |

## What Troy can train on your Mac

| Unified memory | Max model (4-bit QLoRA) |
|---|---|
| 8 GB | ~1.5B |
| 16 GB | ~4B |
| 24 GB | ~8B |
| 36 GB | ~14B |
| 64 GB | ~32B |
| 128 GB | ~70B |

## DPO without a second model

DPO normally keeps a frozen reference copy of the model in memory. Troy
zeroes the LoRA scales to recover the reference model from the policy model
itself — no second copy, which matters on unified memory.

## Data formats

Auto-detected from the first record: Alpaca (`instruction`/`output`),
ShareGPT (`conversations`), chat (`messages`), `prompt`/`completion`,
plain `text`, and preference pairs (`prompt`/`chosen`/`rejected`) for DPO.
Files: `.jsonl`, `.json`, `.csv`.

## License

Apache-2.0
