# Examples

Runnable configs + data for each cookbook recipe — every one verified end-to-end
on an M1 Pro (16 GB). Full walkthroughs with real outputs: https://gettroy.app/cookbook/

| Example | Task | Shows |
|---|---|---|
| [`persona-bot/`](persona-bot/) | SFT | Style/persona transfer from 20 examples |
| [`json-extractor/`](json-extractor/) | SFT (completions) | Text → structured JSON |
| [`concise-dpo/`](concise-dpo/) | DPO | Preference tuning (no reference copy in memory) |
| [`concise-orpo/`](concise-orpo/) | ORPO | Preference tuning with no reference model at all |
| [`ollama-gguf/`](ollama-gguf/) | SFT + export | Llama-arch base → GGUF → Ollama |

Run any of them:

```bash
cd examples/persona-bot
troy train
troy chat -p "My phone won't charge, what should I do?"
```
