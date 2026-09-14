# travel-tools — fine-tune for tool calling

Teaches `Qwen3-1.7B-4bit` to call travel-planner tools reliably:
`search_flights`, `find_hotels`, `get_weather`, `add_to_itinerary`.

**Why 1.7B and not 0.6B?** We tried 0.6B first, three times. It learned the
tool-call *format* perfectly (val loss 0.06) but never the *decision*: it
refused valid requests, invented departure cities instead of asking, and
swapped behaviors between prompts on every retrain. The conditional rule
"origin present → call the tool; origin absent → ask first" reliably emerges
one size up. If your task is format-only (always call, fixed schema), 0.6B is
fine; if the model must decide *whether* and *what to ask*, start at 1.7B
(~1 GB at 4-bit — still an easy iPhone fit).

Small models botch tool calls — wrong argument names, invented tools,
malformed JSON. This is one of the highest-leverage fine-tunes a small
on-device model can get.

The dataset is **chat format plus a `tools` key** per record. mlx-lm feeds
`tools` to the model's chat template, so the schemas render in training
exactly as they will at inference. Assistant turns emit Qwen-native
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks; tool
results come back as `role: "tool"` messages. It also includes no-tool
examples so the model learns when *not* to call.

**Memory needed to train:** the ~1 GB figure above is *inference* on the
phone; training on the Mac needs more — 4-bit QLoRA holds the quantized
weights plus LoRA gradients, optimizer state, and activations for
`seq_len: 2048`. Measured: **5.2 GB peak** for this config (`batch_size: 1`,
429 records, 16 GB M-series Mac) — so an 8 GB Mac handles it. `troy train` prints
`Peak mem` as it runs — if you climb past your machine, drop `seq_len` or set
`grad_checkpoint: true`.

```bash
cd examples/travel-tools
troy train
troy chat -p "Find me a flight from Delhi to Jaipur on October 2nd."
# expect: <tool_call>{"name": "search_flights", ...}</tool_call>
```

Run it on your iPhone with the matching demo app
([`examples/ios/TroyTravel`](../ios/TroyTravel/)):

```bash
troy export -f ios
```

All flight/hotel/weather data in the dataset and the app is mock — this is a
tool-calling demo, not a booking engine.
