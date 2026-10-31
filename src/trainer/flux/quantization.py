"""Weight-only quantization for the Flux transformer and large text encoders.

* ``fp8``: per-tensor scaled ``float8_e4m3fn`` storage, dequantized to the compute dtype on
  every forward (same scheme as OneTrainer's ``LinearFp8``). Works for any ``nn.Linear`` and
  is PEFT-LoRA compatible because the class subclasses ``nn.Linear``.
* ``nf4``: bitsandbytes ``Linear4bit`` (NF4). Requires CUDA at load time.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

FP8_DTYPE = torch.float8_e4m3fn


class Fp8Linear(nn.Linear):
    """``nn.Linear`` whose weight is stored as float8_e4m3fn with a per-tensor scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.detach().to(self.compute_dtype) * self.scale.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        return F.linear(x.to(self.compute_dtype), weight, bias)


def quantize_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp8_weight, scale)`` such that ``fp8_weight * scale ~= weight``."""
    weight = weight.to(torch.float32)
    abs_max = weight.abs().max()
    scale = torch.clamp(abs_max, min=1e-12) / torch.finfo(FP8_DTYPE).max
    return (weight / scale).to(FP8_DTYPE), scale.to(torch.float32)


def linear_module_names(model: nn.Module, predicate: Callable[[str], bool]) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and predicate(name)
    ]


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new_module)


def swap_linears_to_fp8(model: nn.Module, names: list[str], compute_dtype: torch.dtype) -> None:
    """Replace the named ``nn.Linear`` modules with ``Fp8Linear`` (parameters stay on meta)."""
    for name in names:
        old = model.get_submodule(name)
        new = Fp8Linear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            device=old.weight.device,
            compute_dtype=compute_dtype,
        )
        new.weight = nn.Parameter(
            torch.empty_like(old.weight, dtype=FP8_DTYPE, device=old.weight.device),
            requires_grad=False,
        )
        _set_submodule(model, name, new)


def quantize_state_dict_fp8(
    state_dict: dict[str, torch.Tensor],
    names: list[str],
    *,
    compute_dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """In place: fp8-quantize ``<name>.weight`` for each name, add ``<name>.scale``; cast the rest."""
    quantized_keys = {f"{name}.weight" for name in names}
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if key in quantized_keys:
            q, scale = quantize_to_fp8(tensor.to(device))
            state_dict[key] = q
            state_dict[key[: -len(".weight")] + ".scale"] = scale.to(device)
        elif tensor.is_floating_point():
            state_dict[key] = tensor.to(device=device, dtype=compute_dtype)


def swap_linears_to_nf4(
    model: nn.Module,
    names: list[str],
    *,
    compute_dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """Replace loaded (CPU) Linear layers with bitsandbytes NF4 layers, quantizing on ``device``."""
    import bitsandbytes as bnb  # noqa: PLC0415

    if torch.device(device).type != "cuda":
        raise RuntimeError("nf4 quantization requires a CUDA device")
    for name in names:
        old = model.get_submodule(name)
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            compute_dtype=compute_dtype,
            quant_type="nf4",
            device="cpu",
        )
        new.weight = bnb.nn.Params4bit(
            old.weight.data.to(dtype=compute_dtype, device="cpu"),
            requires_grad=False,
            quant_type="nf4",
        )
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.to(dtype=compute_dtype), requires_grad=False)
        new = new.to(device)  # quantizes the weight on CUDA
        _set_submodule(model, name, new)
        del old
