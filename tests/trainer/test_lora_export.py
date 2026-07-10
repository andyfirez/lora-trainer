"""Tests for kohya LoRA export/import."""

import torch
import torch.nn as nn
from src.trainer.config import TrainConfig
from src.trainer.sdxl.lora_export import (
    _peft_param_to_kohya_keys,
    apply_kohya_state_dict,
    apply_lora_metadata_to_config,
    detect_lora_format,
    export_kohya_state_dict,
    infer_kohya_lora_metadata,
)


class _FakePeftModule(nn.Module):
    def __init__(self, param_names: list[str]) -> None:
        super().__init__()
        for name in param_names:
            parts = name.split(".")
            module: nn.Module = self
            for part in parts[:-1]:
                if not hasattr(module, part):
                    setattr(module, part, nn.Module())
                module = getattr(module, part)
            setattr(module, parts[-1], nn.Parameter(torch.randn(2, 2)))


def test_peft_param_to_kohya_keys_unet_to_q() -> None:
    name = "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight"
    kohya_base, suffix = _peft_param_to_kohya_keys(name, "lora_unet_")
    assert kohya_base == "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"
    assert suffix == "lora_down.weight"


def test_peft_param_to_kohya_keys_unet_to_out() -> None:
    name = "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight"
    kohya_base, suffix = _peft_param_to_kohya_keys(name, "lora_unet_")
    assert kohya_base == "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0"
    assert suffix == "lora_up.weight"


def test_peft_param_to_kohya_keys_te1_q_proj() -> None:
    name = "base_model.model.text_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight"
    kohya_base, suffix = _peft_param_to_kohya_keys(name, "lora_te1_")
    assert kohya_base == "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
    assert suffix == "lora_down.weight"


def test_detect_lora_format() -> None:
    assert detect_lora_format({"lora_unet_x.lora_down.weight": torch.tensor(1.0)}) == "kohya"
    assert detect_lora_format({"lora_unet_base_model_model_x_lora_A_default": torch.tensor(1.0)}) == "legacy"


def test_export_kohya_state_dict_contains_alpha_and_down_up() -> None:
    unet = _FakePeftModule([
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    ])
    config = TrainConfig(lora_alpha=8.0)
    state_dict = export_kohya_state_dict(unet, nn.Module(), nn.Module(), config)

    base = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    assert f"{base}.lora_down.weight" in state_dict
    assert f"{base}.lora_up.weight" in state_dict
    assert f"{base}.alpha" in state_dict
    assert state_dict[f"{base}.alpha"].item() == 8.0


def test_kohya_export_apply_roundtrip() -> None:
    param_names = [
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    ]
    source = _FakePeftModule(param_names)
    target = _FakePeftModule(param_names)
    config = TrainConfig(lora_alpha=16.0)

    exported = export_kohya_state_dict(source, nn.Module(), nn.Module(), config)
    apply_kohya_state_dict(exported, unet=target, text_encoder_1=nn.Module(), text_encoder_2=nn.Module(), config=config)

    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        source.named_parameters(),
        target.named_parameters(),
        strict=True,
    ):
        assert src_name == tgt_name
        assert torch.allclose(src_param, tgt_param)


def test_export_kohya_state_dict_ff_module() -> None:
    unet = _FakePeftModule([
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_A.default.weight",
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_B.default.weight",
    ])
    config = TrainConfig(lora_alpha=32.0)
    state_dict = export_kohya_state_dict(unet, nn.Module(), nn.Module(), config)

    base = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2"
    assert f"{base}.lora_down.weight" in state_dict
    assert f"{base}.lora_up.weight" in state_dict
    assert f"{base}.alpha" in state_dict


def test_infer_kohya_lora_metadata_from_state_dict() -> None:
    state_dict = {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": torch.randn(16, 4),
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": torch.randn(8, 16),
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha": torch.tensor(16.0),
    }

    metadata = infer_kohya_lora_metadata(state_dict)

    assert metadata.rank == 16
    assert metadata.alpha == 16.0
    assert metadata.train_te1 is False
    assert metadata.train_te2 is False


def test_apply_lora_metadata_to_config_overrides_rank() -> None:
    state_dict = {
        "lora_unet_x.lora_down.weight": torch.randn(16, 4),
        "lora_unet_x.lora_up.weight": torch.randn(8, 16),
        "lora_unet_x.alpha": torch.tensor(16.0),
    }
    config = TrainConfig(lora_rank=32, lora_alpha=32.0)

    resolved = apply_lora_metadata_to_config(config, state_dict)

    assert resolved.lora_rank == 16
    assert resolved.lora_alpha == 16.0
