"""Tests for troy push: never overwrite an existing Hub repo without --force."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from troy.push import run_push


@pytest.fixture
def hf_api(monkeypatch):
    api = MagicMock()
    api.whoami.return_value = {"name": "tester"}
    module = types.SimpleNamespace(HfApi=lambda: api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return api


def test_push_refuses_existing_repo(hf_api, tmp_path):
    hf_api.repo_exists.return_value = True
    with pytest.raises(SystemExit, match="--force"):
        run_push(tmp_path, "user/existing")
    hf_api.create_repo.assert_not_called()
    hf_api.upload_folder.assert_not_called()


def test_push_force_overwrites_existing_repo(hf_api, tmp_path):
    hf_api.repo_exists.return_value = True
    url = run_push(tmp_path, "user/existing", force=True)
    hf_api.upload_folder.assert_called_once_with(folder_path=str(tmp_path), repo_id="user/existing")
    assert url == "https://huggingface.co/user/existing"


def test_push_creates_new_repo(hf_api, tmp_path):
    hf_api.repo_exists.return_value = False
    run_push(tmp_path, "user/new", private=False)
    hf_api.create_repo.assert_called_once_with("user/new", exist_ok=True, private=False)
    hf_api.upload_folder.assert_called_once()
