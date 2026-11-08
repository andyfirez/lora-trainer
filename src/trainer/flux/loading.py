"""Load Flux.1 / FLUX.2 Klein components from local single-file ``.safetensors`` checkpoints.

Model configs are derived from the weights themselves (and, for text encoders, from the
user-supplied ``config.json`` directory), so nothing is downloaded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn

from src.trainer.config import FluxQuantization
from src.trainer.flux.quantization import (
    linear_module_names,
    quantize_state_dict_fp8,
    swap_linears_to_fp8,
    swap_linears_to_nf4,
)

logger = logging.getLogger(__name__)

_DIFFUSION_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


# --------------------------------------------------------------------------- helpers


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return dict(load_file(str(path), device="cpu"))


def strip_known_prefix(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return state_dict


def extract_by_marker(
    state_dict: dict[str, torch.Tensor],
    marker: str,
) -> dict[str, torch.Tensor]:
    """Strip whatever prefix precedes ``marker`` (e.g. ``text_encoders.clip_l.transformer.``)."""
    for key in state_dict:
        if key.endswith(marker):
            prefix = key[: -len(marker)]
            if not prefix:
                return state_dict
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    raise ValueError(f"Weights file does not look like the expected model (no key ending with '{marker}')")


def _max_block_index(state_dict: dict[str, Any], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = [int(m.group(1)) for key in state_dict if (m := pattern.match(key))]
    if not indices:
        raise ValueError(f"No '{prefix}.N.' blocks found in transformer weights")
    return max(indices) + 1


def build_module_from_state_dict(
    factory: Callable[[], nn.Module],
    state_dict: dict[str, torch.Tensor],
    *,
    quantization: FluxQuantization,
    compute_dtype: torch.dtype,
    device: torch.device | str,
    linear_predicate: Callable[[str], bool] = lambda name: True,
    allowed_missing: tuple[str, ...] = (),
    allow_unexpected: bool = False,
) -> nn.Module:
    """Instantiate on the meta device, optionally quantize, then assign the weights."""
    with init_empty_weights():
        model = factory()

    names: list[str] = []
    if quantization != FluxQuantization.NONE:
        names = linear_module_names(model, linear_predicate)

    if quantization == FluxQuantization.FP8:
        swap_linears_to_fp8(model, names, compute_dtype)
        quantize_state_dict_fp8(state_dict, names, compute_dtype=compute_dtype, device=device)
    else:
        for key in list(state_dict.keys()):
            tensor = state_dict[key]
            if tensor.is_floating_point():
                state_dict[key] = tensor.to(dtype=compute_dtype)

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = [key for key in result.missing_keys if key not in allowed_missing]
    if missing:
        raise ValueError(f"Weights file is missing {len(missing)} keys, e.g. {missing[:5]}")
    if result.unexpected_keys and not allow_unexpected:
        raise ValueError(
            f"Weights file has {len(result.unexpected_keys)} unexpected keys, e.g. {result.unexpected_keys[:5]}"
        )

    if quantization == FluxQuantization.NF4:
        swap_linears_to_nf4(model, names, compute_dtype=compute_dtype, device=device)

    if hasattr(model, "tie_weights"):
        model.tie_weights()  # e.g. T5 encoder.embed_tokens -> shared

    leftover_meta = [name for name, p in model.named_parameters() if p.is_meta]
    if leftover_meta:
        raise ValueError(f"Parameters left uninitialized after load: {leftover_meta[:5]}")
    model.requires_grad_(False)
    return model


def _is_transformer_block_linear(name: str) -> bool:
    return name.startswith(("transformer_blocks.", "single_transformer_blocks."))


# ----------------------------------------------------------------------- transformers


def flux1_transformer_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    x_embedder = state_dict["x_embedder.weight"]
    inner_dim = x_embedder.shape[0]
    return {
        "in_channels": int(x_embedder.shape[1]),
        "num_layers": _max_block_index(state_dict, "transformer_blocks"),
        "num_single_layers": _max_block_index(state_dict, "single_transformer_blocks"),
        "attention_head_dim": 128,
        "num_attention_heads": int(inner_dim // 128),
        "joint_attention_dim": int(state_dict["context_embedder.weight"].shape[1]),
        "pooled_projection_dim": int(
            state_dict["time_text_embed.text_embedder.linear_1.weight"].shape[1]
        ),
        "guidance_embeds": any("guidance_embedder" in key for key in state_dict),
    }


def flux2_transformer_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    x_embedder = state_dict["x_embedder.weight"]
    inner_dim = int(x_embedder.shape[0])
    ff_in = state_dict["transformer_blocks.0.ff.linear_in.weight"]
    return {
        "in_channels": int(x_embedder.shape[1]),
        "num_layers": _max_block_index(state_dict, "transformer_blocks"),
        "num_single_layers": _max_block_index(state_dict, "single_transformer_blocks"),
        "attention_head_dim": 128,
        "num_attention_heads": inner_dim // 128,
        "joint_attention_dim": int(state_dict["context_embedder.weight"].shape[1]),
        "mlp_ratio": float(ff_in.shape[0]) / (2.0 * inner_dim),
        "guidance_embeds": any("guidance_embedder" in key for key in state_dict),
    }


def load_flux1_transformer(
    path: Path,
    *,
    quantization: FluxQuantization,
    compute_dtype: torch.dtype,
    device: torch.device | str,
) -> nn.Module:
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders.single_file_utils import (
        convert_flux_transformer_checkpoint_to_diffusers,
    )

    state_dict = strip_known_prefix(load_safetensors(path), _DIFFUSION_PREFIXES)
    if any(key.startswith("double_blocks.") for key in state_dict):
        state_dict = convert_flux_transformer_checkpoint_to_diffusers(state_dict)
    kwargs = flux1_transformer_kwargs(state_dict)
    logger.info("Flux.1 transformer config from weights: %s", kwargs)
    return build_module_from_state_dict(
        lambda: FluxTransformer2DModel(**kwargs),
        state_dict,
        quantization=quantization,
        compute_dtype=compute_dtype,
        device=device,
        linear_predicate=_is_transformer_block_linear,
    )


def load_flux2_transformer(
    path: Path,
    *,
    quantization: FluxQuantization,
    compute_dtype: torch.dtype,
    device: torch.device | str,
) -> nn.Module:
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )

    state_dict = strip_known_prefix(load_safetensors(path), _DIFFUSION_PREFIXES)
    if any(key.endswith((".comfy_quant", ".weight_scale")) for key in state_dict):
        state_dict = dequantize_comfy_state_dict(state_dict, compute_dtype)
    if any(key.startswith("double_blocks.") for key in state_dict):
        state_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
    kwargs = flux2_transformer_kwargs(state_dict)
    logger.info("FLUX.2 transformer config from weights: %s", kwargs)
    return build_module_from_state_dict(
        lambda: Flux2Transformer2DModel(**kwargs),
        state_dict,
        quantization=quantization,
        compute_dtype=compute_dtype,
        device=device,
        linear_predicate=_is_transformer_block_linear,
    )


# ------------------------------------------------------------------------------- VAEs

_FLUX1_VAE_CONFIG: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "latent_channels": 16,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "force_upcast": True,
}


def _vae_state_dict_to_diffusers(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if any(key.startswith("encoder.down_blocks.") for key in state_dict):
        return state_dict
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    converted = convert_ldm_vae_checkpoint(dict(state_dict), config)
    for key, value in state_dict.items():
        if key.startswith("bn."):  # FLUX.2 VAE batch-norm latent statistics
            converted[key] = value
    return converted


def _load_vae(model: nn.Module, state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> nn.Module:
    state_dict = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]
    if missing:
        raise ValueError(f"VAE weights are missing {len(missing)} keys, e.g. {missing[:5]}")
    model.requires_grad_(False)
    return model.to(dtype=dtype).eval()


def load_flux1_vae(path: Path, *, dtype: torch.dtype) -> nn.Module:
    from diffusers import AutoencoderKL

    state_dict = strip_known_prefix(load_safetensors(path), ("first_stage_model.",))
    converted = _vae_state_dict_to_diffusers(state_dict, _FLUX1_VAE_CONFIG)
    model = AutoencoderKL(**_FLUX1_VAE_CONFIG)
    return _load_vae(model, converted, dtype)


def _infer_vae_channels(state_dict: dict[str, torch.Tensor], part: str) -> tuple[int, ...] | None:
    """Per-level output channels of the VAE ``encoder``/``decoder`` in block_out_channels order."""
    ldm_name = "down" if part == "encoder" else "up"
    diff_name = "down_blocks" if part == "encoder" else "up_blocks"
    ldm = re.compile(rf"^{part}\.{ldm_name}\.(\d+)\.block\.0\.conv2\.weight$")
    diff = re.compile(rf"^{part}\.{diff_name}\.(\d+)\.resnets\.0\.conv2\.weight$")
    found: dict[int, int] = {}
    for key, value in state_dict.items():
        for pattern in (ldm, diff):
            match = pattern.match(key)
            if match:
                found[int(match.group(1))] = int(value.shape[0])
    if not found or sorted(found) != list(range(len(found))):
        return None
    channels = tuple(found[i] for i in range(len(found)))
    # Diffusers stores decoder levels deepest-first, so reverse to config order.
    if part == "decoder" and any(diff.match(k) for k in state_dict):
        channels = tuple(reversed(channels))
    return channels


def load_flux2_vae(path: Path, *, dtype: torch.dtype) -> nn.Module:
    from diffusers import AutoencoderKLFlux2

    state_dict = strip_known_prefix(load_safetensors(path), ("first_stage_model.",))
    # Some exports nest the quant convs under the encoder/decoder module.
    state_dict = {
        (key.split(".", 1)[1] if key.startswith(("encoder.quant_conv.", "decoder.post_quant_conv.")) else key): value
        for key, value in state_dict.items()
    }
    config = {
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
    }
    converted = _vae_state_dict_to_diffusers(state_dict, config)
    kwargs: dict[str, Any] = {}
    encoder_channels = _infer_vae_channels(converted, "encoder")
    decoder_channels = _infer_vae_channels(converted, "decoder")
    if encoder_channels is not None and len(encoder_channels) == 4:
        kwargs["block_out_channels"] = encoder_channels
    if decoder_channels is not None and len(decoder_channels) == 4:
        kwargs["decoder_block_out_channels"] = decoder_channels
    model = AutoencoderKLFlux2(**kwargs)
    return _load_vae(model, converted, dtype)


# ---------------------------------------------------------------------- text encoders


def load_tokenizer(tokenizer_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def load_clip_l(
    weights_path: Path,
    config_dir: Path,
    *,
    compute_dtype: torch.dtype,
) -> nn.Module:
    from transformers import CLIPTextConfig, CLIPTextModel

    config = CLIPTextConfig.from_pretrained(str(config_dir))
    state_dict = extract_by_marker(
        load_safetensors(weights_path), "text_model.embeddings.token_embedding.weight"
    )
    state_dict = {k: v for k, v in state_dict.items() if k.startswith("text_model.")}
    return build_module_from_state_dict(
        lambda: CLIPTextModel(config),
        state_dict,
        quantization=FluxQuantization.NONE,
        compute_dtype=compute_dtype,
        device="cpu",
        allowed_missing=("text_model.embeddings.position_ids",),
        allow_unexpected=True,
    ).eval()


def load_t5(
    weights_path: Path,
    config_dir: Path,
    *,
    compute_dtype: torch.dtype,
    quantization: FluxQuantization,
) -> nn.Module:
    from transformers import T5Config, T5EncoderModel

    config = T5Config.from_pretrained(str(config_dir))
    state_dict = extract_by_marker(
        load_safetensors(weights_path), "encoder.block.0.layer.0.SelfAttention.q.weight"
    )
    # Drop scaled-fp8 bookkeeping entries; they are not parameters of the module.
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(("scale_weight", "scaled_fp8"))}
    # Large text encoders are fp8-quantized whenever any quantization is requested (nf4 is
    # transformer-only because it needs CUDA at load time).
    te_quantization = FluxQuantization.FP8 if quantization != FluxQuantization.NONE else FluxQuantization.NONE
    model = build_module_from_state_dict(
        lambda: T5EncoderModel(config),
        state_dict,
        quantization=te_quantization,
        compute_dtype=compute_dtype,
        device="cpu",
        linear_predicate=lambda name: name.startswith("encoder.block"),
        allowed_missing=("encoder.embed_tokens.weight",),
        allow_unexpected=True,
    )
    return model.eval()


_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _unswizzle_block_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Reverse the cuBLAS tiled layout of NVFP4 block scales (see comfy_kitchen.float_utils)."""
    n_row_blocks = -(-rows // 128)
    n_col_blocks = -(-cols // 4)
    step = blocked.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    step = step.reshape(n_row_blocks, n_col_blocks, 4, 32, 4).reshape(n_row_blocks, n_col_blocks, 128, 4)
    full = step.permute(0, 2, 1, 3).reshape(n_row_blocks * 128, n_col_blocks * 4)
    return full[:rows, :cols]


def _dequantize_nvfp4(weight: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor, dtype: torch.dtype):
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32)
    lo = (weight & 0x0F).long()
    hi = (weight >> 4).long()
    values = lut[torch.stack([hi, lo], dim=-1).reshape(weight.shape[0], -1)]
    rows, cols = values.shape
    scales = _unswizzle_block_scales(block_scale, rows, cols // 16).to(torch.float32) * global_scale.float()
    return (values.reshape(rows, -1, 16) * scales.unsqueeze(-1)).reshape(rows, cols).to(dtype)


def dequantize_comfy_state_dict(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Dequantize ComfyUI mixed-precision checkpoints (fp8 per-tensor and nvfp4 layers).

    Tensors without ``comfy_quant`` metadata are returned unchanged; helper keys
    (``weight_scale``, ``weight_scale_2``, ``comfy_quant``, ``input_scale``) are dropped.
    """
    import json

    out: dict[str, torch.Tensor] = {}
    for key in list(state_dict):
        if key.endswith((".weight_scale", ".weight_scale_2", ".comfy_quant", ".input_scale")):
            continue
        tensor = state_dict.pop(key)
        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            meta = state_dict.get(f"{base}.comfy_quant")
            if meta is not None:
                fmt = json.loads(bytes(meta.tolist()).decode("utf-8")).get("format")
            elif f"{base}.weight_scale" in state_dict and tensor.dtype == torch.float8_e4m3fn:
                fmt = "float8_e4m3fn"
            else:
                fmt = None
            if fmt == "nvfp4":
                tensor = _dequantize_nvfp4(
                    tensor, state_dict[f"{base}.weight_scale"], state_dict[f"{base}.weight_scale_2"], dtype
                )
            elif fmt == "float8_e4m3fn":
                scale = state_dict[f"{base}.weight_scale"].float()
                tensor = tensor.to(torch.float32).mul_(scale).to(dtype)
            elif fmt is not None:
                raise ValueError(f"Unsupported quantization format '{fmt}' for {key}")
        out[key] = tensor
    return out


def _align_qwen3_config_with_weights(config, state_dict: dict) -> None:
    """Override size fields of ``config`` with what the checkpoint actually contains.

    The bundled config may describe a different Qwen3 size (e.g. 4B) than the weights
    (e.g. 8B used by Flux2 Klein 9B). Only output dims are used, so packed/quantized
    input dims do not matter.
    """
    layer_ids = {
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("layers.") and k.split(".")[1].isdigit()
    }
    if layer_ids:
        config.num_hidden_layers = max(layer_ids) + 1
    config.hidden_size = int(state_dict["embed_tokens.weight"].shape[1])
    gate = state_dict.get("layers.0.mlp.gate_proj.weight")
    if gate is not None:
        config.intermediate_size = int(gate.shape[0])
    q_norm = state_dict.get("layers.0.self_attn.q_norm.weight")
    q_proj = state_dict.get("layers.0.self_attn.q_proj.weight")
    k_proj = state_dict.get("layers.0.self_attn.k_proj.weight")
    if q_norm is not None:
        head_dim = int(q_norm.shape[0])
        config.head_dim = head_dim
        if q_proj is not None:
            config.num_attention_heads = int(q_proj.shape[0]) // head_dim
        if k_proj is not None:
            config.num_key_value_heads = int(k_proj.shape[0]) // head_dim
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = ["full_attention"] * config.num_hidden_layers


def load_qwen3(
    weights_path: Path,
    config_dir: Path,
    *,
    compute_dtype: torch.dtype,
    quantization: FluxQuantization,
) -> nn.Module:
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config.from_pretrained(str(config_dir))
    raw = load_safetensors(weights_path)
    state_dict = strip_known_prefix(raw, ("model.",)) if any(k.startswith("model.") for k in raw) else raw
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head.")}
    if any(k.endswith((".comfy_quant", ".weight_scale")) for k in state_dict):
        state_dict = dequantize_comfy_state_dict(state_dict, compute_dtype)
    if "embed_tokens.weight" not in state_dict:
        raise ValueError("Qwen3 weights file does not contain 'embed_tokens.weight'")
    _align_qwen3_config_with_weights(config, state_dict)
    te_quantization = FluxQuantization.FP8 if quantization != FluxQuantization.NONE else FluxQuantization.NONE
    model = build_module_from_state_dict(
        lambda: Qwen3Model(config),
        state_dict,
        quantization=te_quantization,
        compute_dtype=compute_dtype,
        device="cpu",
        linear_predicate=lambda name: name.startswith("layers."),
        allow_unexpected=True,
    )
    return model.eval()
