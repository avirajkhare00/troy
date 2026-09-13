"""Tests for the iOS export validation."""

import json

import pytest

from troy.export import _export_ios


def _make_model(tmp_path, model_type="qwen3", size_mb=400, quant=True, tokenizer=True):
    cfg = {"model_type": model_type}
    if quant:
        cfg["quantization"] = {"bits": 4, "group_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.seek(size_mb * 1_000_000 - 1)
        f.write(b"\0")
    if tokenizer:
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
    return tmp_path


def test_ios_export_ok(tmp_path, capsys):
    _export_ios(_make_model(tmp_path))
    out = capsys.readouterr().out
    assert "comfortable" in out
    assert (tmp_path / "README-iOS.md").exists()


def test_ios_export_large_model_needs_entitlement(tmp_path, capsys):
    _export_ios(_make_model(tmp_path, model_type="llama", size_mb=3600))
    assert "Increased Memory Limit" in capsys.readouterr().out


def test_ios_export_unsupported_arch_fails(tmp_path):
    with pytest.raises(SystemExit):
        _export_ios(_make_model(tmp_path, model_type="not_a_real_arch"))


def test_ios_export_missing_tokenizer_fails(tmp_path):
    with pytest.raises(SystemExit):
        _export_ios(_make_model(tmp_path, tokenizer=False))


def test_base_is_quantized_local_dir(tmp_path):
    from troy.export import _base_is_quantized

    q = tmp_path / "quant"; q.mkdir()
    (q / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization": {"bits": 4}}))
    f = tmp_path / "full"; f.mkdir()
    (f / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert _base_is_quantized(str(q)) is True
    assert _base_is_quantized(str(f)) is False
