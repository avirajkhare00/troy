# travel-tools — fine-tune for tool calling

Teaches `Qwen3-0.6B-4bit` to call travel-planner tools reliably:
`search_flights`, `find_hotels`, `get_weather`, `add_to_itinerary`.

Small models botch tool calls — wrong argument names, invented tools,
malformed JSON. This is one of the highest-leverage fine-tunes a small
on-device model can get.

The dataset is **chat format plus a `tools` key** per record. mlx-lm feeds
`tools` to the model's chat template, so the schemas render in training
exactly as they will at inference. Assistant turns emit Qwen-native
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks; tool
results come back as `role: "tool"` messages. It also includes no-tool
examples so the model learns when *not* to call.

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
