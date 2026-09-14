# Troy command reference

## troy init
`troy init [--template chat|dpo|orpo] [--path troy.yaml] [--force]`
Writes config + sample data (data/train.jsonl or data/preferences.jsonl).

## troy doctor
No args. Prints chip, unified memory, macOS, Metal/MLX status, comfortable
model size, and reminds that any mlx-lm architecture is supported.

## troy train
`troy train [-c troy.yaml]`
Dispatches on config: task sft/dpo/orpo; `data.train` folder → vision SFT.
Adapter lands at `<output>/adapter/adapters.safetensors` (+ adapter_config.json).

## troy chat
`troy chat [-c troy.yaml] [--model REPO] [--base-only] [-p PROMPT] [--image IMG] [--tools schemas.json] [--system PROMPT] [--max-tokens N] [--temperature T]`
- Default: REPL with the config's base + trained adapter (warns if no adapter).
- `-p`: one-shot. `--image` (vision models): one-shot only, requires `-p`.
- `--tools`: JSON list of OpenAI-style function specs, rendered into the chat
  template — REQUIRED to see `<tool_call>`s from a tool-tuned model (without
  it the model correctly refuses, since it sees no tools). Pair with
  `--system` set to the system prompt used in training. Not with --image.
- REPL commands: `/exit`, `/clear` (keeps the --system message).

## troy eval
`troy eval [-c troy.yaml] [-p PROMPT]... [--max-tokens N]`
SFT only for the loss table: base vs tuned val loss + perplexity (base is
recovered by zeroing LoRA scales — single model load). `-p` is repeatable;
each prompt generates base-vs-tuned side by side at temperature 0.

## troy serve
`troy serve [-c troy.yaml] [--model REPO] [--base-only] [--host 127.0.0.1] [--port 8080] [--max-tokens N]`
OpenAI-compatible: POST /v1/chat/completions, GET /v1/models. Response JSON
puts reasoning-model thinking in `message.reasoning`; final text in
`message.content` (give reasoning models max_tokens headroom). A `tools`
array in the request body is rendered into the chat template (mlx-lm server
passthrough), so tool-tuned models emit `<tool_call>`s over the API too.

## troy export
`troy export [-c troy.yaml] [-f mlx|gguf|ios] [--save-path DIR] [--dequantize]`
Fuses adapter into base → `<output>/fused/`. gguf additionally writes
`ggml-model-f16.gguf` (llama/mistral/mixtral archs, unquantized base).
Ollama: `FROM ./output/fused/ggml-model-f16.gguf` in a Modelfile, then
`ollama create name -f Modelfile`.
ios fuses SAFELY for quantized bases: dequantize-fuse then requantize at
8 bits. Fusing a LoRA straight into 4-bit weights silently erases the
adapter (deltas drown in quantization noise — the export behaves like the
base model); 8-bit requantization preserves tuned behavior, 6-bit already
degrades it. It then validates the result for mlx-swift-lm
(supported model_type, tokenizer.json present, weight size vs iPhone RAM)
and writes `README-iOS.md` with the Swift loading snippet and the memory
entitlements needed (Increased Memory Limit for >~2 GB of weights). For iOS,
train from a 4-bit base and keep weights under ~4 GB; skip --dequantize.

## troy push
`troy push USER/REPO [-c troy.yaml] [--fused] [--public]`
Uploads `<output>/adapter` (default) or `<output>/fused` (--fused).
Private by default. Requires `hf auth login` or HF_TOKEN.

## troy data
`troy data inspect PATH` — records, detected format, record sizes.

`troy data validate PATH` — lints the dataset: invalid JSON lines, mixed
formats, empty required fields, chat records with no assistant message,
unknown sharegpt roles, chosen==rejected, exact duplicates. Exit 1 on issues.

`troy data synth [--from PATH] [--seed "TASK"] [--n 100] [-f chat|preference|tools] [--tools schemas.json] [--no-think] [--teacher auto|REPO] [-o FILE] [--max-tokens 2048] [--temperature 0.8]`
Synthesizes a dataset with a local mlx-lm teacher. Needs --from and/or --seed.
`-f tools` generates tool-calling scenarios: requires --tools (JSON list of
OpenAI-style function specs) and --seed (becomes the system prompt). Scenarios
mix direct calls, multi-call chains, clarify-first (missing required arg), and
no-tool answers; assistant turns carry <think> reasoning traces (omit with
--no-think). Training WITH traces is what teaches a small model to think and
STOP — an adapter trained on empty thinks loops forever if thinking is enabled
at inference.
--from: file or folder, chunked (~4000 chars) and cycled; examples are
grounded in the chunks. --teacher auto picks by unified memory
(8→Qwen3-1.7B, 16→4B, 24→8B, 36→14B, 64→30B-A3B, all 4-bit instruct).
Dedupes on the prompt, strips think-blocks, refuses to overwrite output.

## troy mesh
`troy mesh serve [--from PATH] [--seed "TASK"] [--n 100] [-f chat|preference|tools] [--tools schemas.json] [--no-think] [-o FILE] [--max-tokens 2048] [--temperature 0.8] [--host 0.0.0.0] [--port 8765] [--token T] [--lease-timeout 300] [--linger 30]`
Coordinates distributed synthesis: renders the same prompts as `troy data
synth`, hands them to LAN workers over HTTP (bearer token; GET /v1/work,
POST /v1/results, GET /v1/status), validates/dedupes results identically,
writes the same output file. The coordinator never loads a model. Unanswered
leases requeue after --lease-timeout; after the target is hit it lingers
--linger seconds for in-flight results. Same --from/--seed/--tools rules as
synth; refuses to overwrite output.

`troy mesh join URL --token T [--model auto|REPO] [--name NAME] [--batch 2]`
Runs the teacher on this Mac against a coordinator. --model auto sizes to
this machine's memory (workers can differ). iPhone equivalent: the
`examples/ios/TroyWorker` app — enter URL + token; foreground-only (lease
requeue covers backgrounding).

## Memory sizing (4-bit QLoRA guidance, not a catalog)

| Unified memory | Comfortable base |
|---|---|
| 8 GB | ~1.5B |
| 16 GB | ~4B |
| 24 GB | ~8B |
| 36 GB | ~14B |
| 64 GB | ~32B |
| 128 GB | ~70B |

## iOS: run a Troy model on iPhone/iPad

Pipeline: `troy train` → `troy export -f ios` → load in an app via
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm). Working example
apps: `examples/ios/TroyChat` (plain chat) and `examples/ios/TroyTravel`
(tool calling; pairs with the `examples/travel-tools` dataset).

Swift side (mlx-swift-lm v3+, needs swift-huggingface + swift-transformers):
```swift
import MLXLLM; import MLXLMCommon; import MLXHuggingFace
import HuggingFace; import Tokenizers
// bundled folder:
let m = try await loadModelContainer(from: dirURL, using: #huggingFaceTokenizerLoader())
// or from the Hub (after `troy push you/model --fused`):
let m = try await loadModelContainer(from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(), configuration: .init(id: "you/model"))
let session = ChatSession(m)                       // tools: / toolDispatch: for tool calling
for try await chunk in session.streamResponse(to: prompt) { ... }
```

Hard-won gotchas (all hit in practice):
- Physical device only — MLX kernels need a real GPU, the simulator has none.
- Xcode 26: run `xcodebuild -downloadPlatform iOS` and
  `xcodebuild -downloadComponent MetalToolchain` once, or device builds fail.
- Swift macros need trust: "Trust & Enable" in Xcode, or
  `xcodebuild -skipMacroValidation` on the CLI.
- Phone needs Developer Mode on (Settings → Privacy & Security) and pairing
  (`xcrun devicectl manage pair`); build destinations use the Xcode UDID
  (from `-showdestinations`), not the devicectl identifier.
- Free personal teams cannot sign the Extended Virtual Addressing
  entitlement; Increased Memory Limit is only needed over ~2 GB of weights.
- Bundle a model for dev as a *folder reference* named `TroyModel`
  (`optional: true` + `buildPhase: resources` in XcodeGen); ship real apps
  with an on-demand download instead.
- Set `MLX.Memory.cacheLimit` small (~20 MB) on iOS — cached buffers count
  against the app's jetsam limit.
