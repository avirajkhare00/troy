"""Config and sample-data templates for `troy init`."""

CHAT_TEMPLATE = """\
# Troy config — supervised fine-tuning (SFT)
# Run with: troy train

base: mlx-community/Qwen3-0.6B-4bit   # any MLX model on the HF Hub, or a local path
task: sft

data:
  train: ./data/train.jsonl   # alpaca, sharegpt, chat, completions, or text — auto-detected
  format: auto
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto        # sized from your Mac's unified memory
  seq_len: 2048
  lora:
    r: 8
    alpha: 16

output: ./output
"""

DPO_TEMPLATE = """\
# Troy config — preference tuning (DPO)
# Run with: troy train

base: mlx-community/Qwen3-0.6B-4bit
task: dpo

data:
  train: ./data/preferences.jsonl   # {"prompt": ..., "chosen": ..., "rejected": ...}
  format: preference
  val_split: 0.05

training:
  epochs: 1
  lr: 5e-6
  batch_size: auto
  seq_len: 2048
  lora:
    r: 8
    alpha: 16
  dpo:
    beta: 0.1

output: ./output
"""

SAMPLE_SFT_DATA = [
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "output": "Endless blue expanse\nWaves whisper against the shore\nSalt hangs in the air",
    },
    {
        "instruction": "Explain what a LoRA adapter is in one sentence.",
        "output": "A LoRA adapter is a small set of low-rank matrices trained alongside a frozen model so it can be specialized cheaply.",
    },
]

SAMPLE_DPO_DATA = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "I think it might be Lyon, but I'm not sure.",
    },
    {
        "prompt": "Summarize photosynthesis in one sentence.",
        "chosen": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "rejected": "Plants eat sunlight.",
    },
]

TEMPLATES = {
    "chat": (CHAT_TEMPLATE, "train.jsonl", SAMPLE_SFT_DATA),
    "dpo": (DPO_TEMPLATE, "preferences.jsonl", SAMPLE_DPO_DATA),
}
