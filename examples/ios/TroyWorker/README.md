# TroyWorker — turn an iPhone into a mesh worker

A SwiftUI app that joins a [`troy mesh`](https://gettroy.app) — your Mac
coordinates, idle iPhones generate synthetic training data with an on-device
teacher model via [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).
The phones make the dataset, the Mac trains on it, the phones can run the
result.

On the Mac:

```bash
troy mesh serve --from ./docs --n 500        # prints the URL + token to join
```

On each iPhone: open TroyWorker, enter the URL and token, load a teacher
model, tap **Start working**. Records accumulate in the Mac's `train.jsonl`,
validated identically to `troy data synth`. Any Mac can join too:
`troy mesh join http://<mac-ip>:8765 --token <token>`.

## Build

Requires Xcode 16+:

```bash
cd examples/ios/TroyWorker
open TroyWorker.xcodeproj
```

(The project is generated from `project.yml` — if you change it, regenerate
with [XcodeGen](https://github.com/yonaskolb/XcodeGen): `xcodegen generate`.
Note the `info:` block: ATS's `NSAllowsLocalNetworking` and the local-network
usage description are dict-valued Info.plist keys, so this target uses a
generated plist instead of `GENERATE_INFOPLIST_FILE`.)

Set your development team under Signing & Capabilities, then run on a
**physical device** (MLX needs a real GPU; the simulator won't work).
When Xcode asks about the `MLXHuggingFaceMacros` macro package, choose
"Trust & Enable" (from the command line: `xcodebuild -skipMacroValidation …`).

## Teacher model

The default is `mlx-community/Qwen3-4B-4bit` (~2.3 GB, good for iPhone 15 Pro
and later — the Increased Memory Limit entitlement is preconfigured). On older
phones use `mlx-community/Qwen3-1.7B-4bit`. The app downloads from the Hub
once and caches; a bundled `TroyModel` folder reference works too, exactly as
in TroyChat.

## How it behaves

- Pulls one work item at a time (a fully rendered teacher prompt — the phone
  never sees your source documents' chunking or tool schemas), generates,
  POSTs the raw text back, repeats until the coordinator says done.
- The screen stays awake while working; keep the app foregrounded and the
  phone on a charger. Backgrounding stops the worker — the coordinator
  requeues anything unfinished after the lease timeout, so nothing is lost.
- Duplicate or malformed generations are rejected Mac-side; the counters show
  both completed items and actually-accepted records.

iOS 17+, iPhone and iPad.

Built by [Aviraj](https://aviraj.dev).
