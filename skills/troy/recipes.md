# Verified Troy recipes (summaries)

All verified end-to-end on an M1 Pro, 16 GB. Full pages: https://gettroy.app/cookbook/
Runnable configs + data: `examples/` in github.com/avirajkhare00/troy

## 01 Persona bot (sft, alpaca)
~20 `{"instruction","output"}` pairs in a fixed voice → model answers unseen
questions in that voice. epochs 8, lr 5e-5. More data / bigger base = tighter style.

## 02 Text→JSON extractor (sft, completions)
`{"prompt": "Extract ... <text>", "completion": "<json>"}` ~50 rows.
epochs 4, lr 5e-5, temperature 0 at inference. Generalizes to unseen entities.
Verified eval: val ppl 27.5 (base) → 1.3 (tuned).

## 03 Concise answers (dpo, preference)
Pairs: chosen = short answer, rejected = rambling. iters 30, lr 1e-5, beta 0.1.
Verified: reward acc 1.0; base rambles, tuned answers in two lines.

## 04 Local OpenAI API (serve)
`troy serve` then POST /v1/chat/completions. Adapter loads automatically from
config. Reasoning models need max_tokens ~600+.

## 05 Ship to Ollama (sft + export -f gguf)
Use a llama-arch base (e.g. mlx-community/SmolLM2-135M-Instruct). After
training: `troy export -f gguf`, Modelfile `FROM ./output/fused/ggml-model-f16.gguf`,
`ollama create`.

## 06 ORPO (orpo, preference)
Same data as 03, `task: orpo`, `orpo: {lambda: 0.2}`, lr 2e-5. Verified 3.2 it/s
vs DPO's 1.3–2.0 on identical data; no reference model in the math.

## 07 Vision (sft, image folder) — needs troy-cli[vision]
`data.train: ./data` where the folder holds images + metadata.jsonl
`{"file_name","question","answer"}`. base mlx-community/Qwen2-VL-2B-Instruct-4bit,
batch 1, lr 1e-4. Verified: 2.4 GB peak; correct answers on unseen images via
`troy chat --image`.

## 08 Docs → dataset → model (data synth + sft)
No hand-written data: `troy data synth --from ./README.md --seed "answering
questions about <project>" --n 100`, spot-check + `troy data validate`, then
`troy train`. Verified on 16 GB (teacher Qwen3-4B-4bit): grounded Q&A pairs,
0 validation issues. Preference variant: `-f preference` → `task: orpo`.
