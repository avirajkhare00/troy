from troy.hardware import auto_batch_size, detect, model_guidance


def test_detect_runs():
    hw = detect()
    assert hw.memory_gb > 0
    assert hw.arch


def test_model_guidance_tiers():
    assert "70B" in model_guidance(128)
    assert "4B" in model_guidance(16)
    assert "small" in model_guidance(4)


def test_auto_batch_size_bounds():
    for mem in (8, 16, 64, 128):
        for seq in (512, 2048, 8192):
            bs = auto_batch_size(mem, seq)
            assert 1 <= bs <= 8


def test_auto_batch_size_scales_down_with_seq_len():
    assert auto_batch_size(16, 8192) <= auto_batch_size(16, 512)
