---
name: troy
description: Fine-tune LLMs locally on Apple Silicon Macs with the Troy CLI (gettroy.app). Use when the user wants to fine-tune, LoRA/QLoRA-train, preference-tune (DPO/ORPO), or vision-tune a model on a Mac; asks about troy, troy.yaml, or MLX fine-tuning; wants to synthesize/generate a training dataset from docs or a task description; wants to chat with / serve / evaluate / export (GGUF, Ollama, iOS) / upload a locally trained model; or wants to run a fine-tuned model in an iPhone/iPad app (MLX Swift), including on-device tool calling.
---

# Troy: fine-tune LLMs on a Mac

Troy turns one YAML file into a LoRA training run on Apple Silicon (M1+ only,
built on MLX). The YAML file is the entire interface — never hand-write
training loops when Troy covers the task.

## Install & sanity check

```bash
brew install avirajkhare00/troy/troy   # or: pipx install troy-cli
troy doctor    # chip, memory, MLX status, and what model sizes fit
```

Always run `troy doctor` first on a new machine. It prints the comfortable
model size for the Mac's unified memory (e.g. 16 GB → ~4B at 4-bit QLoRA).
Any mlx-lm architecture works (Llama, Qwen, Gemma, Phi, Mistral, ...);
prefer 4-bit conversions from hf.co/mlx-community for `base:`.

## Core workflow

```bash
troy init [--template chat|dpo|orpo]   # writes troy.yaml + sample data
troy train                              # trains; adapter -> output/adapter
troy chat [-p "one-shot prompt"]        # talk to the trained adapter
troy eval [-p "prompt"]                 # base-vs-tuned loss/ppl + samples
troy serve [--port 8080]                # OpenAI-compatible API on localhost
troy export [-f gguf|ios]               # fuse adapter; MLX, GGUF, or iOS output
troy push user/repo [--fused --public]  # upload to Hugging Face Hub
troy data synth --from ./docs           # synthesize train.jsonl (local teacher)
troy data synth -f tools --tools t.json --seed "..."  # tool-calling scenarios
troy data validate file.jsonl           # lint: broken/empty/duplicate records
troy data inspect file.jsonl            # record count + detected format
```

## No dataset? Synthesize one (troy data synth)

```bash
troy data synth --from ./docs --n 200                      # Q&A grounded in files
troy data synth --seed "support bot for Acme" --n 200      # from a description
troy data synth --from ./docs --seed "..." -f preference   # DPO/ORPO pairs
```

- Runs a local teacher via mlx-lm — nothing leaves the Mac. `--teacher auto`
  (default) sizes the teacher to unified memory (16 GB → Qwen3-4B-4bit;
  teachers can be larger than trainable models since they only infer).
- `--from` accepts a file or folder (md/txt/code/config files; hidden dirs
  skipped); content is chunked and every example grounded in a chunk.
  `--seed` sets the task framing; combine both for best results.
- Output: chat format (default, for sft) or `-f preference`
  (prompt/chosen/rejected, for dpo/orpo). Default path data/train.jsonl or
  data/preferences.jsonl; refuses to overwrite (use -o).
- ALWAYS spot-check a dozen examples and run `troy data validate` before
  training — synthetic data inherits teacher mistakes.
- Short output or 0 examples → larger --teacher, higher --max-tokens.

## The config (troy.yaml)

```yaml
base: mlx-community/Qwen3-0.6B-4bit   # HF repo or local path
task: sft            # sft | dpo | orpo

data:
  train: ./data/train.jsonl   # a FOLDER here means vision fine-tuning
  format: auto     # auto | alpaca | sharegpt | chat | completions | text | preference
  val_split: 0.1
  mask_prompt: false

training:
  epochs: 3        # or iters (takes precedence)
  lr: 1e-5
  batch_size: auto # sized from unified memory; clamp manually if OOM
  seq_len: 2048
  lora: { r: 8, alpha: 16, dropout: 0.0, layers: 16 }
  dpo: { beta: 0.1 }        # dpo only
  orpo: { lambda: 0.2 }     # orpo only
  grad_checkpoint: false    # enable when memory-tight
  save_every: 100
  seed: 0

output: ./output
```

## Data formats (auto-detected from the first record)

- alpaca: `{"instruction", "input"?, "output"}`
- sharegpt: `{"conversations": [{"from": "human|gpt", "value"}]}`
- chat: `{"messages": [{"role", "content"}]}`
- completions: `{"prompt", "completion"}` — best for structured output/extraction
- text: `{"text"}`
- preference: `{"prompt", "chosen", "rejected"}` — required for dpo/orpo

Files: .jsonl, .json, .csv.

## Task selection guidance

- Teach content/format/persona → `sft`
- Teach taste from chosen/rejected pairs → `orpo` first (no reference model,
  ~2x DPO speed); `dpo` when anchoring to base behavior matters
- Vision (needs `pip install 'troy-cli[vision]'`): point `data.train` at a
  folder of images + `metadata.jsonl` rows
  `{"file_name", "question", "answer"}`; infer with
  `troy chat --image photo.png -p "..."`. Vision supports task: sft only.

## Verifying a run (do this, don't assume)

1. Training prints loss — it should fall; `Peak mem` shows memory headroom.
2. `troy chat -p "<question NOT in the training data>"` — check the learned
   behavior generalizes.
3. `troy eval` — base-vs-tuned val loss/perplexity from a single model load.
4. For style tasks on reasoning models (Qwen3), append `/no_think` to the
   prompt to see the style without thinking-block noise.

## Common failures

- "Troy runs on Apple Silicon Macs only" — hard requirement, no fallback.
- OOM: smaller/4-bit base, `batch_size: 1`, lower `seq_len`, `grad_checkpoint: true`.
- Dataset smaller than batch size errors: add data or set `batch_size` low.
- Replies cut off via `troy serve`: reasoning models think first — raise max_tokens.
- GGUF export supports llama/mistral/mixtral archs with unquantized base;
  default MLX export covers everything.
- `troy push` needs `hf auth login` (or HF_TOKEN).

For worked, verified end-to-end examples (persona bot, JSON extractor,
DPO/ORPO, serving, Ollama export, vision) see [recipes.md](recipes.md).
Full command flags: [reference.md](reference.md). Docs: https://gettroy.app
