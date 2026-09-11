# Contributing to Troy

Thanks for helping. Two principles shape every contribution:

1. **The tool stays light.** Troy is a thin contract: YAML in, trained model out.
   New CLI surface needs a strong reason; most ideas belong in the cookbook instead.
2. **The cookbook is verified.** A recipe must be run end-to-end on real Apple
   Silicon hardware before it's written down. Include the hardware you ran it on
   and the actual output you saw. No hypothetical recipes.

## Development setup

```bash
git clone https://github.com/avirajkhare00/troy && cd troy/cli
python3 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest tests/
```

## Pull requests

- CI runs unit tests plus real SFT/DPO/export smoke tests on Apple Silicon
  runners — it must be green.
- Bug fixes: include a test that fails without the fix when practical.
- Recipes: add the page to `web/cookbook/` following the existing format
  (data shape → troy.yaml → verify commands → real output).

## Releases (maintainers)

Tag `vX.Y.Z` and push — CI builds, publishes to PyPI, creates the GitHub
release, and bumps the Homebrew formula automatically.
