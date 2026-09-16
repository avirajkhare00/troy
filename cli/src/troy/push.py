"""Upload a trained adapter or fused model to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path


def run_push(folder: Path, repo: str, private: bool = True, force: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        raise SystemExit(
            "Not logged in to Hugging Face. Run: hf auth login "
            "(or set HF_TOKEN) and retry."
        )
    if api.repo_exists(repo) and not force:
        raise SystemExit(
            f"Hub repo {repo} already exists. Re-run with --force to overwrite "
            "its files, or choose a different repo id."
        )
    print(f"Logged in as {who['name']}. Uploading {folder} -> {repo} "
          f"({'private' if private else 'public'}) ...")
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_folder(folder_path=str(folder), repo_id=repo)
    url = f"https://huggingface.co/{repo}"
    print(f"Done: {url}")
    return url
