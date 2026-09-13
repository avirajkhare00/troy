# TroyChat — run your Troy model on iPhone

A minimal SwiftUI chat app that runs a model you fine-tuned with
[Troy](https://gettroy.app) **entirely on-device**, using
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

Train on your Mac, chat on your iPhone:

```bash
troy train
troy export -f ios          # fused model + iPhone-fit check
```

## Build

Requires Xcode 16+:

```bash
cd examples/ios/TroyChat
open TroyChat.xcodeproj
```

(The project is generated from `project.yml` — if you change it, regenerate
with [XcodeGen](https://github.com/yonaskolb/XcodeGen): `xcodegen generate`.)

Set your development team under Signing & Capabilities, then run on a
**physical device** (MLX needs a real GPU; the simulator won't work).
When Xcode asks about the `MLXHuggingFaceMacros` macro package, choose
"Trust & Enable" (from the command line: `xcodebuild -skipMacroValidation …`).

## Get your model into the app

Two options:

1. **Bundle it (quick local test)** — drag the `troy export -f ios` output
   folder into the Xcode project as a *folder reference* (blue folder) named
   `TroyModel`. The app loads it at launch. Fine for development; don't ship
   multi-GB bundles to the App Store.
2. **Download from the Hub** — `troy push you/your-model --fused`, then enter
   `you/your-model` in the app. The app downloads once and caches. (Private
   repos need a token — simplest is to make the repo public, or bundle instead.)

## Notes

- The `Increased Memory Limit` and `Extended Virtual Addressing` entitlements
  are already configured — needed for models over ~2 GB of weights.
- Train from a 4-bit base (e.g. `mlx-community/Qwen3-0.6B-4bit`) and keep
  weights under ~4 GB for 8 GB-RAM iPhones; `troy export -f ios` tells you
  where you stand.
- iOS 17+, iPhone and iPad.

Built by [Aviraj](https://aviraj.dev).
