"""CPU tests on tiny random Flux models: shapes, LoRA targets, export keys, resume, quantization."""

import logging

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLFlux2,
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
)
from peft import get_peft_model

from src.trainer.base_trainer import TrainContext
from src.trainer.config import FluxQuantization, TrainConfig
from src.trainer.flux import flow
from src.trainer.flux.loading import (
    build_module_from_state_dict,
    flux1_transformer_kwargs,
    flux2_transformer_kwargs,
)
from src.trainer.flux.lora import (
    FLUX1_TRANSFORMER_TARGETS,
    FLUX2_TRANSFORMER_TARGETS,
    apply_flux_resume_state,
    collect_flux_resume_state,
    export_flux_lora_state_dict,
)
from src.trainer.flux.quantization import Fp8Linear
from src.trainer.flux1.trainer import Flux1LoRATrainer
from src.trainer.flux2.trainer import Flux2KleinLoRATrainer
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config


def _tiny_flux1(guidance: bool = True) -> FluxTransformer2DModel:
    torch.manual_seed(0)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=24,
        guidance_embeds=guidance,
        axes_dims_rope=(4, 6, 6),
    )


def _tiny_flux2() -> Flux2Transformer2DModel:
    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=48,
        mlp_ratio=3.0,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )


def _lora(model, targets):
    return get_peft_model(
        model,
        build_sdxl_lora_config(rank=4, alpha=2.0, dropout=0.0, target_modules=targets),
    )


def _ctx(config: TrainConfig) -> TrainContext:
    return TrainContext(
        config=config,
        device=torch.device("cpu"),
        weight_dtype=torch.bfloat16,
        vae_dtype="float32",
        attention_mechanism="sdpa",
        resume_state=None,
        log=logging.getLogger("test"),
    )


# ----------------------------------------------------------------------- flow helpers


def test_flux1_pack_roundtrip_and_ids():
    latents = torch.randn(2, 4, 8, 6)
    packed = flow.flux1_pack(latents)
    assert packed.shape == (2, 12, 16)
    assert torch.equal(flow.flux1_unpack(packed, 8, 6), latents)
    ids = flow.flux1_image_ids(8, 6, torch.device("cpu"), torch.float32)
    assert ids.shape == (12, 3)
    assert ids[:, 1].max() == 3 and ids[:, 2].max() == 2


def test_flux2_patchify_and_pack_roundtrip():
    latents = torch.randn(2, 4, 8, 6)
    patched = flow.flux2_patchify(latents)
    assert patched.shape == (2, 16, 4, 3)
    assert torch.equal(flow.flux2_unpatchify(patched), latents)
    packed = flow.flux2_pack(patched)
    assert packed.shape == (2, 12, 16)
    assert torch.equal(flow.flux2_unpack(packed, 4, 3), patched)
    assert flow.flux2_image_ids(2, 4, 3, torch.device("cpu")).shape == (2, 12, 4)
    assert flow.flux2_text_ids(2, 5, torch.device("cpu")).shape == (2, 5, 4)


def test_sigma_sampling_range_and_shift():
    from src.trainer.config import FluxTimestepDistribution

    for dist in FluxTimestepDistribution:
        sigmas = flow.sample_sigmas(256, dist, 3.0, torch.device("cpu"))
        assert sigmas.min() > 0 and sigmas.max() <= 1.0
    # shift > 1 moves mass toward noisier timesteps
    base = torch.tensor([0.5])
    assert flow.shift_sigmas(base, 3.0).item() > 0.5
    assert flow.shift_sigmas(base, 1.0).item() == pytest.approx(0.5)


# -------------------------------------------------------------------- config detection


def test_config_is_derived_from_weights():
    t1 = _tiny_flux1()
    kwargs = flux1_transformer_kwargs(t1.state_dict())
    assert kwargs["num_layers"] == 1 and kwargs["num_single_layers"] == 1
    assert kwargs["guidance_embeds"] is True
    assert kwargs["joint_attention_dim"] == 32 and kwargs["pooled_projection_dim"] == 24
    assert flux1_transformer_kwargs(_tiny_flux1(guidance=False).state_dict())["guidance_embeds"] is False

    k2 = flux2_transformer_kwargs(_tiny_flux2().state_dict())
    assert k2["mlp_ratio"] == pytest.approx(3.0)
    assert k2["joint_attention_dim"] == 48 and k2["guidance_embeds"] is False


# ------------------------------------------------------------------------- LoRA / export


@pytest.mark.parametrize(
    ("factory", "targets", "expected_modules"),
    [
        (_tiny_flux1, FLUX1_TRANSFORMER_TARGETS, 17),
        (_tiny_flux2, FLUX2_TRANSFORMER_TARGETS, 14),
    ],
)
def test_lora_targets_match_expected_modules(factory, targets, expected_modules):
    model = _lora(factory(), targets)
    lora_a = [n for n, _ in model.named_parameters() if "lora_A" in n]
    assert len(lora_a) == expected_modules


def test_export_keys_fold_alpha_and_resume_roundtrip():
    model = _lora(_tiny_flux1(), FLUX1_TRANSFORMER_TARGETS)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.constant_(param, 0.5)
    parts = {"transformer": model}

    state = export_flux_lora_state_dict(parts, rank=4, alpha=2.0)
    assert all(k.startswith("transformer.") for k in state)
    assert not any(k.endswith(".alpha") for k in state)
    b_key = "transformer.transformer_blocks.0.attn.to_q.lora_B.weight"
    a_key = "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
    assert b_key in state and a_key in state
    # alpha / rank = 0.5 folded into lora_B
    assert state[b_key].float().mean().item() == pytest.approx(0.25, abs=1e-2)
    assert state[b_key].dtype == torch.bfloat16

    saved = collect_flux_resume_state(parts)
    fresh = _lora(_tiny_flux1(), FLUX1_TRANSFORMER_TARGETS)
    applied = apply_flux_resume_state({"transformer": fresh}, saved)
    assert applied == len(saved)
    for name, param in fresh.named_parameters():
        if "lora_B" in name:
            assert torch.allclose(param, torch.full_like(param, 0.5))


def test_resume_state_without_matching_weights_raises():
    model = _lora(_tiny_flux1(), FLUX1_TRANSFORMER_TARGETS)
    with pytest.raises(ValueError, match="does not contain"):
        apply_flux_resume_state({"transformer": model}, {})


# --------------------------------------------------------------------------- quantization


def test_fp8_load_matches_reference_output():
    reference = _tiny_flux1()
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    kwargs = flux1_transformer_kwargs(state)
    kwargs["attention_head_dim"] = 16
    kwargs["num_attention_heads"] = 2
    kwargs["axes_dims_rope"] = (4, 6, 6)

    model = build_module_from_state_dict(
        lambda: FluxTransformer2DModel(**kwargs),
        state,
        quantization=FluxQuantization.FP8,
        compute_dtype=torch.float32,
        device="cpu",
        linear_predicate=lambda n: n.startswith(("transformer_blocks.", "single_transformer_blocks.")),
    )
    fp8_layers = [m for m in model.modules() if isinstance(m, Fp8Linear)]
    assert fp8_layers and all(m.weight.dtype == torch.float8_e4m3fn for m in fp8_layers)

    inputs = _flux1_inputs()
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = model(**inputs, return_dict=False)[0]
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert (actual - expected).abs().mean() < 0.1 * expected.abs().mean() + 0.05

    lora_model = _lora(model, FLUX1_TRANSFORMER_TARGETS)  # LoRA on top of fp8 layers
    assert any("lora_A" in n for n, _ in lora_model.named_parameters())


def test_load_rejects_incomplete_weights():
    reference = _tiny_flux1()
    state = dict(reference.state_dict())
    state.pop("x_embedder.weight")
    with pytest.raises(ValueError, match="missing"):
        build_module_from_state_dict(
            lambda: _tiny_flux1(),
            state,
            quantization=FluxQuantization.NONE,
            compute_dtype=torch.float32,
            device="cpu",
        )


def _flux1_inputs(batch: int = 1, h: int = 4, w: int = 4):
    return {
        "hidden_states": torch.randn(batch, (h // 2) * (w // 2), 16),
        "encoder_hidden_states": torch.randn(batch, 5, 32),
        "pooled_projections": torch.randn(batch, 24),
        "timestep": torch.full((batch,), 0.5),
        "img_ids": flow.flux1_image_ids(h, w, torch.device("cpu"), torch.float32),
        "txt_ids": torch.zeros(5, 3),
        "guidance": torch.full((batch,), 1.0),
    }


# ----------------------------------------------------------------- end-to-end loss step


def _tiny_vae1() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        latent_channels=4,
        norm_num_groups=4,
        layers_per_block=1,
        scaling_factor=0.36,
        shift_factor=0.11,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )


def test_flux1_flow_loss_backward_on_cpu():
    config = TrainConfig.model_validate({"model_type": "flux1"})
    trainer = Flux1LoRATrainer(config)
    transformer = _lora(_tiny_flux1(), FLUX1_TRANSFORMER_TARGETS)
    trainer._transformer = transformer
    trainer._forward_transformer = transformer
    trainer._vae = _tiny_vae1()
    trainer._guidance_embeds = True

    latents = torch.randn(2, 4, 8, 8)  # raw latents: 8x8 -> 4x4 packed tokens
    text = (torch.randn(2, 5, 32), torch.randn(2, 24))
    loss = trainer._flow_loss(latents, text, _ctx(config))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for n, p in transformer.named_parameters() if "lora_" in n and p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_flux2_flow_loss_backward_on_cpu():
    config = TrainConfig.model_validate({"model_type": "flux2_klein"})
    trainer = Flux2KleinLoRATrainer(config)
    transformer = _lora(_tiny_flux2(), FLUX2_TRANSFORMER_TARGETS)
    trainer._transformer = transformer
    trainer._forward_transformer = transformer
    trainer._vae = AutoencoderKLFlux2(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        latent_channels=4,
        norm_num_groups=4,
        layers_per_block=1,
    )
    trainer._guidance_embeds = False

    latents = torch.randn(2, 4, 8, 8)
    text = (torch.randn(2, 5, 48),)
    loss = trainer._flow_loss(latents, text, _ctx(config))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for n, p in transformer.named_parameters() if "lora_" in n and p.requires_grad]
    assert grads and all(g is not None for g in grads)
