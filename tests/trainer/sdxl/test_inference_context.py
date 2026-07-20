"""Unit tests for merged-adapter sampling orchestration."""

from unittest.mock import MagicMock, patch

import torch
from src.trainer.config import TrainConfig
from src.trainer.sdxl.inference_context import (
    merge_adapters_for_inference,
    run_merged_adapter_sampling,
    run_sampling_pass_with_embeds,
    unmerge_adapters,
)


def _make_peft_module(base_model: MagicMock | None = None) -> MagicMock:
    module = MagicMock()
    inner = MagicMock()
    inner.model = base_model or MagicMock()
    module.base_model = inner
    return module


def test_merge_and_unmerge_adapters_for_inference() -> None:
    unet = _make_peft_module()
    te1 = _make_peft_module()
    te2 = _make_peft_module()
    config = TrainConfig(text_encoder_1={"train": True}, text_encoder_2={"train": False})

    state = merge_adapters_for_inference(
        unet=unet,
        text_encoder_1=te1,
        text_encoder_2=te2,
        lora_config=config,
        merge_unet=True,
    )

    unet.merge_adapter.assert_called_once()
    te1.merge_adapter.assert_called_once()
    te2.merge_adapter.assert_not_called()
    assert state.models.unet is unet.base_model.model
    assert state.models.text_encoder_1 is te1.base_model.model
    assert state.models.text_encoder_2 is te2

    unmerge_adapters(unet=unet, text_encoder_1=te1, text_encoder_2=te2, state=state)
    unet.unmerge_adapter.assert_called_once()
    te1.unmerge_adapter.assert_called_once()
    te2.unmerge_adapter.assert_not_called()


@patch("src.trainer.sdxl.inference_context.run_sampling_pass_with_embeds")
def test_run_merged_adapter_sampling_delegates_to_pass_helper(mock_pass: MagicMock) -> None:
    unet = _make_peft_module()
    te1 = _make_peft_module()
    te2 = _make_peft_module()
    vae = MagicMock()
    config = TrainConfig()

    run_merged_adapter_sampling(
        unet=unet,
        text_encoder_1=te1,
        text_encoder_2=te2,
        vae=vae,
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        noise_scheduler=MagicMock(),
        lora_config=config,
        sampling_config=config,
        device=torch.device("cpu"),
        sample_prompts=["a cat"],
        output_dir=MagicMock(),
        output_stem="test",
        log=MagicMock(),
        merge_unet=False,
    )

    mock_pass.assert_called_once()
    unet.unmerge_adapter.assert_not_called()
    te1.unmerge_adapter.assert_not_called()


@patch("src.trainer.sdxl.inference_context.build_inference_scheduler")
@patch("src.trainer.sdxl.inference_context.run_sdxl_sampling_pass")
@patch("src.trainer.sdxl.inference_context.precompute_all_sample_embeds", return_value=[])
@patch("src.trainer.sdxl.inference_context.SDXLSamplingSession.create")
def test_run_sampling_pass_moves_text_encoders_to_device_before_embeds(
    mock_session_create: MagicMock,
    mock_precompute: MagicMock,
    mock_pass: MagicMock,
    mock_build_scheduler: MagicMock,
) -> None:
    te1 = MagicMock(name="te1")
    te2 = MagicMock(name="te2")
    te1.to.side_effect = lambda *args, **kwargs: te1
    te2.to.side_effect = lambda *args, **kwargs: te2
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_sampling_pass_with_embeds(
        inference_unet=MagicMock(),
        inference_te1=te1,
        inference_te2=te2,
        vae=MagicMock(),
        tokenizer_1=MagicMock(),
        tokenizer_2=MagicMock(),
        noise_scheduler=MagicMock(),
        sampling_config=config,
        device=device,
        sample_prompts=["a cat"],
        output_dir=MagicMock(),
        output_stem="test",
        log=MagicMock(),
    )

    assert te1.to.call_args_list[0] == (( ), {"device": device, "dtype": torch.float16})
    assert te2.to.call_args_list[0] == (( ), {"device": device, "dtype": torch.float16})
    mock_precompute.assert_called_once()
    _, precompute_kwargs = mock_precompute.call_args
    assert precompute_kwargs["text_encoder_1"] is te1
    assert precompute_kwargs["text_encoder_2"] is te2
