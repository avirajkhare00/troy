# TroyTravel — on-device tool calling, demoed as a trip planner

A SwiftUI demo where a model you fine-tuned with [Troy](https://gettroy.app)
plans a trip **on your iPhone** by calling tools: `search_flights`,
`find_hotels`, `get_weather`, and `add_to_itinerary`. Watch the itinerary
panel fill up as the model calls tools.

**All tool data is mock.** This is a demo of on-device *tool calling* — the
thing small models are bad at until you fine-tune them — not a booking app.

## Why fine-tune for this?

Small models (0.6–4B) running on a phone routinely botch tool calls: wrong
argument names, invented tools, malformed JSON. Teaching tool-call emission is
one of the highest-leverage fine-tunes a small model can get. Train with Troy
on your Mac, then run it here:

```bash
troy train                  # tool-calling dataset, see examples/
troy export -f ios
```

Any chat model in the supported architectures works untuned too (Qwen3 bases
handle basic tool calls) — the point of the demo is comparing base vs tuned.

## Build

```bash
cd examples/ios/TroyTravel
open TroyTravel.xcodeproj   # generated from project.yml (xcodegen generate)
```

Set your team under Signing & Capabilities and run on a **physical device**
(MLX needs a real GPU). Choose "Trust & Enable" when Xcode asks about the
`MLXHuggingFaceMacros` macro package. Get your model in the same two ways as TroyChat:
bundle a `TroyModel` folder reference, or enter a Hub repo id
(`troy push you/your-model --fused`).

The `Increased Memory Limit` / `Extended Virtual Addressing` entitlements are
preconfigured; keep weights under ~4 GB (`troy export -f ios` checks this).

Built by [Aviraj](https://aviraj.dev).
