import pytest
import yaml

from troy.config import TroyConfig, load_config


def make(tmp_path, raw):
    p = tmp_path / "troy.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


BASE = {"base": "mlx-community/Qwen3-0.6B-4bit", "data": {"train": "./data/train.jsonl"}}


def test_minimal_config(tmp_path):
    cfg = load_config(make(tmp_path, BASE))
    assert cfg.task == "sft"
    assert cfg.training.batch_size == "auto"
    assert cfg.training.lora.r == 8
    assert cfg.output_path.name == "output"


def test_lora_scale_is_alpha_over_r():
    cfg = TroyConfig.model_validate({**BASE, "training": {"lora": {"r": 16, "alpha": 32}}})
    assert cfg.training.lora.scale == 2.0


def test_dpo_defaults():
    cfg = TroyConfig.model_validate({**BASE, "task": "dpo"})
    assert cfg.training.dpo.beta == 0.1


def test_rejects_unknown_task():
    with pytest.raises(ValueError):
        TroyConfig.model_validate({**BASE, "task": "rlhf"})


def test_rejects_empty_base():
    with pytest.raises(ValueError):
        TroyConfig.model_validate({"base": "  ", "data": {"train": "x.jsonl"}})


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="troy init"):
        load_config(tmp_path / "nope.yaml")


def test_adapter_path_under_output():
    cfg = TroyConfig.model_validate({**BASE, "output": "/tmp/run1"})
    assert str(cfg.adapter_path) == "/tmp/run1/adapter"


def test_orpo_config():
    cfg = TroyConfig.model_validate(
        {**BASE, "task": "orpo", "training": {"orpo": {"lambda": 0.2}}}
    )
    assert cfg.task == "orpo"
    assert cfg.training.orpo.lam == 0.2
