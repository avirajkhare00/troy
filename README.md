# Troy

**Fine-tune LLMs on your MacBook with one YAML file.**

![troy train, then chat with the result — real session on an M1 Pro](web/assets/troy-demo.gif)

Troy is a command-line tool for fine-tuning and preference-tuning language
models locally on Apple Silicon. No CUDA, no cloud, no training pipeline —
write a config, run one command, and train on the machine you already own.

Built on [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm). Unified memory means a
36 GB MacBook fine-tunes models that need a workstation GPU anywhere else.

**Website:** https://avirajkhare00.github.io/troy/

## Quickstart

```bash
brew install avirajkhare00/troy/troy   # or: pipx install troy-cli

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

## What Troy can train on your Mac

| Unified memory | Max model (4-bit QLoRA) |
|---|---|
| 8 GB | ~1.5B |
| 16 GB | ~4B |
| 24 GB | ~8B |
| 36 GB | ~14B |
| 64 GB | ~32B |
| 128 GB | ~70B |

## Repository layout

- [`cli/`](cli/) — the Troy CLI (Python, MLX)
- [`web/`](web/) — the website, deployed to GitHub Pages from `main`

## License

Apache-2.0
