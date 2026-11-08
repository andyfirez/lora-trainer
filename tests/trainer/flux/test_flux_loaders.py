"""Loaders for text encoders and VAEs from local single files (tiny random models)."""

import pytest
import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, Qwen3Config, Qwen3Model, T5Config, T5EncoderModel

from src.trainer.config import FluxQuantization
from src.trainer.flux.loading import (
    load_clip_l,
    load_flux1_vae,
    load_flux2_vae,
    load_qwen3,
    load_t5,
)
from src.trainer.flux.quantization import Fp8Linear


def _contiguous(state: dict) -> dict:
    return {k: v.contiguous() for k, v in state.items()}


def test_load_clip_l_with_comfy_prefix(tmp_path):
    config = CLIPTextConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, vocab_size=64, max_position_embeddings=8,
    )
    config.save_pretrained(tmp_path)
    reference = CLIPTextModel(config).eval()
    state = {f"text_encoders.clip_l.transformer.{k}": v for k, v in reference.state_dict().items()}
    path = tmp_path / "clip.safetensors"
    save_file(_contiguous(state), str(path))

    model = load_clip_l(path, tmp_path, compute_dtype=torch.float32)
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        expected = reference(ids).pooler_output
        actual = model(ids).pooler_output
    assert torch.allclose(expected, actual, atol=1e-5)


def test_load_t5_fp8_quantizes_blocks_and_keeps_embeddings(tmp_path):
    config = T5Config(
        vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2,
        feed_forward_proj="gated-gelu",
    )
    config.save_pretrained(tmp_path)
    reference = T5EncoderModel(config).eval()
    state = {k: v for k, v in reference.state_dict().items() if k != "encoder.embed_tokens.weight"}
    path = tmp_path / "t5.safetensors"
    save_file(_contiguous(state), str(path))

    model = load_t5(path, tmp_path, compute_dtype=torch.float32, quantization=FluxQuantization.FP8)
    assert any(isinstance(m, Fp8Linear) for m in model.encoder.block.modules())
    assert not isinstance(model.shared, Fp8Linear)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out = model(ids)[0]
    assert out.shape == (1, 6, 16) and torch.isfinite(out).all()


def test_load_qwen3_strips_model_prefix_and_drops_lm_head(tmp_path):
    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    reference = Qwen3Model(config).eval()
    state = {f"model.{k}": v for k, v in reference.state_dict().items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    path = tmp_path / "qwen.safetensors"
    save_file(_contiguous(state), str(path))

    model = load_qwen3(path, tmp_path, compute_dtype=torch.float32, quantization=FluxQuantization.NONE)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        expected = reference(ids, output_hidden_states=True).hidden_states[-2]
        actual = model(ids, output_hidden_states=True).hidden_states[-2]
    assert torch.allclose(expected, actual, atol=1e-5)


def test_load_qwen3_rejects_wrong_file(tmp_path):
    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )
    config.save_pretrained(tmp_path)
    path = tmp_path / "bad.safetensors"
    save_file({"something.weight": torch.zeros(2, 2)}, str(path))
    with pytest.raises(ValueError):
        load_qwen3(path, tmp_path, compute_dtype=torch.float32, quantization=FluxQuantization.NONE)


def test_flux2_vae_loads_diffusers_format_checkpoint(tmp_path):
    from diffusers import AutoencoderKLFlux2

    reference = AutoencoderKLFlux2()
    path = tmp_path / "vae.safetensors"
    save_file(_contiguous(reference.state_dict()), str(path))
    model = load_flux2_vae(path, dtype=torch.float32)
    assert model.config.latent_channels == reference.config.latent_channels
    assert torch.equal(model.bn.running_mean, reference.bn.running_mean)


def test_flux1_vae_loads_diffusers_format_checkpoint(tmp_path):
    from diffusers import AutoencoderKL

    from src.trainer.flux.loading import _FLUX1_VAE_CONFIG

    reference = AutoencoderKL(**_FLUX1_VAE_CONFIG)
    path = tmp_path / "ae.safetensors"
    save_file(_contiguous(reference.state_dict()), str(path))
    model = load_flux1_vae(path, dtype=torch.float32)
    assert model.config.latent_channels == 16
    assert model.config.shift_factor == pytest.approx(0.1159)


def test_dequantize_comfy_fp8_and_nvfp4():
    import json

    from src.trainer.flux.loading import dequantize_comfy_state_dict

    def meta(fmt: str) -> torch.Tensor:
        return torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)

    fp8_w = torch.full((4, 4), 2.0).to(torch.float8_e4m3fn)
    packed = torch.full((16, 8), 0x22, dtype=torch.uint8)  # two e2m1 values of 1.0 per byte
    state = {
        "a.weight": fp8_w,
        "a.weight_scale": torch.tensor(0.5),
        "a.input_scale": torch.tensor(1.0),
        "a.comfy_quant": meta("float8_e4m3fn"),
        "b.weight": packed,
        "b.weight_scale": torch.ones(128, 4).to(torch.float8_e4m3fn),
        "b.weight_scale_2": torch.tensor(2.0),
        "b.comfy_quant": meta("nvfp4"),
        "norm.weight": torch.ones(3),
    }
    out = dequantize_comfy_state_dict(state, torch.float32)
    assert set(out) == {"a.weight", "b.weight", "norm.weight"}
    assert torch.allclose(out["a.weight"], torch.ones(4, 4))
    assert out["b.weight"].shape == (16, 16)
    assert torch.allclose(out["b.weight"], torch.full((16, 16), 2.0))


def test_infer_vae_channels_ldm_layout():
    from src.trainer.flux.loading import _infer_vae_channels

    state = {f"decoder.up.{i}.block.0.conv2.weight": torch.zeros(c, c, 3, 3) for i, c in enumerate((96, 192, 384, 384))}
    assert _infer_vae_channels(state, "decoder") == (96, 192, 384, 384)
