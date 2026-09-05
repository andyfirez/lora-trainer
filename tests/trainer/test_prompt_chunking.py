"""Tests for CLIP prompt chunking and chunked SDXL encoding."""

from __future__ import annotations

from typing import Any

import torch
from src.trainer.sdxl.prompt_chunking import (
    CLIP_CONTENT_LENGTH,
    CLIP_SEQUENCE_LENGTH,
    PromptChunkCounts,
    align_embed_pair,
    build_prompt_chunks,
    count_prompt_chunks,
    pack_content_tokens,
)
from src.trainer.sdxl.prompt_encoding import encode_sdxl_prompt_chunked
from src.trainer.sdxl.sampling import PromptEmbedCache, precompute_all_sample_embeds


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    model_max_length = CLIP_SEQUENCE_LENGTH

    def __init__(self, encode_map: dict[str, list[int]] | None = None, comma_id: int = 99) -> None:
        self._encode_map = encode_map or {}
        self._comma_id = comma_id

    def get_vocab(self) -> dict[str, int]:
        return {",</w>": self._comma_id}

    def __call__(self, text: str, truncation: bool = False, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        if text in self._encode_map:
            return {"input_ids": list(self._encode_map[text])}
        if not text:
            return {"input_ids": []}
        return {"input_ids": [10 + index for index, _word in enumerate(text.split())]}


class FakeEncoderOut:
    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor) -> None:
        self.hidden_states = (hidden * 0.5, hidden)
        self._pooled = pooled

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self._pooled
        raise IndexError


class FakeTextEncoder:
    def __init__(self, dim: int, *, pooled: bool = False) -> None:
        self.dim = dim
        self.pooled = pooled

    def __call__(self, input_ids: torch.Tensor, output_hidden_states: bool = True) -> FakeEncoderOut:
        chunk_count, seq = input_ids.shape
        hidden = torch.zeros(chunk_count, seq, self.dim)
        for index in range(chunk_count):
            hidden[index] = float(index + 1)
        pooled = torch.full((chunk_count, 4), 0.0)
        for index in range(chunk_count):
            pooled[index] = float(100 + index)
        return FakeEncoderOut(hidden, pooled)


def test_short_prompt_is_one_77_token_chunk() -> None:
    tokenizer = FakeTokenizer({"cat": [11, 12, 13]})
    chunks = build_prompt_chunks("cat", tokenizer)
    assert len(chunks) == 1
    assert len(chunks[0]) == CLIP_SEQUENCE_LENGTH
    assert chunks[0][0] == tokenizer.bos_token_id
    assert chunks[0][1:4] == [11, 12, 13]
    assert chunks[0][-1] == tokenizer.eos_token_id


def test_empty_prompt_still_encodes_one_chunk() -> None:
    tokenizer = FakeTokenizer()
    chunks = build_prompt_chunks("", tokenizer)
    assert len(chunks) == 1
    assert chunks[0] == [1] + [2] * (CLIP_CONTENT_LENGTH + 1)


def test_long_prompt_splits_into_two_chunks() -> None:
    tokens = list(range(100, 180))
    tokenizer = FakeTokenizer({"long": tokens})
    chunks = build_prompt_chunks("long", tokenizer)
    assert len(chunks) == 2
    assert chunks[0][1:76] == tokens[:75]
    assert chunks[1][1:6] == tokens[75:]


def test_break_forces_a_new_chunk_on_short_text() -> None:
    tokenizer = FakeTokenizer({"hello": [11], "world": [12]})
    chunks = build_prompt_chunks("hello BREAK world", tokenizer)
    assert len(chunks) == 2
    assert chunks[0][1] == 11
    assert chunks[1][1] == 12


def test_comma_backtrack_keeps_tag_together() -> None:
    tokens = [10] * 70 + [99] + [20] * 10
    packed = pack_content_tokens(tokens, bos=1, eos=2, comma_id=99)
    assert len(packed) == 2
    assert packed[0][1:72] == [10] * 70 + [99]
    assert packed[1][1:11] == [20] * 10


def test_count_prompt_chunks_uses_max_of_both_tokenizers() -> None:
    tokenizer_1 = FakeTokenizer({"x": list(range(80))})
    tokenizer_2 = FakeTokenizer({"x": list(range(10))})
    counts = count_prompt_chunks("x", tokenizer_1, tokenizer_2)
    assert counts == PromptChunkCounts(tokens=80, chunks=2)


def test_encode_sdxl_prompt_chunked_concatenates_and_keeps_first_pooled() -> None:
    tokenizer = FakeTokenizer({"long": list(range(80))})
    embeds, pooled = encode_sdxl_prompt_chunked(
        "long",
        tokenizer,
        tokenizer,
        FakeTextEncoder(8),
        FakeTextEncoder(16, pooled=True),
        torch.device("cpu"),
        torch.float32,
        clip_skip=1,
    )
    assert embeds.shape == (1, CLIP_SEQUENCE_LENGTH * 2, 24)
    assert torch.equal(embeds[0, :CLIP_SEQUENCE_LENGTH, 0], torch.ones(CLIP_SEQUENCE_LENGTH))
    assert torch.equal(embeds[0, CLIP_SEQUENCE_LENGTH:, 0], torch.full((CLIP_SEQUENCE_LENGTH,), 2.0))
    assert torch.equal(pooled, torch.full((1, 4), 100.0))


def test_align_embed_pair_pads_shorter_side() -> None:
    empty = torch.full((1, CLIP_SEQUENCE_LENGTH, 3), 7.0)
    prompt = torch.zeros(1, CLIP_SEQUENCE_LENGTH * 2, 3)
    negative = torch.ones(1, CLIP_SEQUENCE_LENGTH, 3)
    aligned_prompt, aligned_negative = align_embed_pair(prompt, negative, empty)
    assert aligned_prompt.shape[1] == aligned_negative.shape[1] == CLIP_SEQUENCE_LENGTH * 2
    assert torch.equal(aligned_negative[0, CLIP_SEQUENCE_LENGTH:], empty[0])


def test_precompute_aligns_positive_and_negative_chunk_counts() -> None:
    tokenizer = FakeTokenizer({"long": list(range(80)), "short": [11]})
    cache = PromptEmbedCache()
    results = precompute_all_sample_embeds(
        sample_prompts=["long"],
        negative_prompt="short",
        tokenizer_1=tokenizer,
        tokenizer_2=tokenizer,
        text_encoder_1=FakeTextEncoder(8),
        text_encoder_2=FakeTextEncoder(16, pooled=True),
        device=torch.device("cpu"),
        dtype=torch.float32,
        clip_skip=1,
        cache=cache,
    )
    assert len(results) == 1
    assert results[0].prompt_embeds.shape[1] == results[0].negative_prompt_embeds.shape[1]
    assert results[0].prompt_embeds.shape[1] == CLIP_SEQUENCE_LENGTH * 2
