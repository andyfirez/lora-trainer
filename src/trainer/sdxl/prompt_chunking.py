"""reForge-compatible CLIP prompt chunking for SDXL sampling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

CLIP_CONTENT_LENGTH = 75
CLIP_SEQUENCE_LENGTH = 77
COMMA_PADDING_BACKTRACK = 20
BREAK_RE = re.compile(r"\s*\bBREAK\b\s*", re.S)
_COMMA_TOKEN_KEYS = (",</w>", ",", "Ġ,")


@dataclass(frozen=True)
class PromptChunkCounts:
    tokens: int
    chunks: int


def _special_token_ids(tokenizer: Any) -> tuple[int, int]:
    eos = int(tokenizer.eos_token_id)
    bos = tokenizer.bos_token_id
    return (int(bos) if bos is not None else eos), eos


def comma_token_id(tokenizer: Any) -> int | None:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for key in _COMMA_TOKEN_KEYS:
        token_id = vocab.get(key)
        if token_id is not None:
            return int(token_id)
    return None


def tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, truncation=False, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        return [int(token) for token in ids[0]]
    return [int(token) for token in ids]


def wrap_chunk(content: list[int], *, bos: int, eos: int) -> list[int]:
    padded = content + [eos] * (CLIP_CONTENT_LENGTH - len(content))
    return [bos, *padded, eos]


def empty_chunk_ids(tokenizer: Any) -> list[int]:
    bos, eos = _special_token_ids(tokenizer)
    return wrap_chunk([], bos=bos, eos=eos)


def pack_content_tokens(
    tokens: list[int],
    *,
    bos: int,
    eos: int,
    comma_id: int | None,
    backtrack: int = COMMA_PADDING_BACKTRACK,
) -> list[list[int]]:
    """Pack content tokens into 77-length CLIP windows."""
    chunks: list[list[int]] = []
    current: list[int] = []
    last_comma = -1

    def flush() -> None:
        nonlocal current, last_comma
        chunks.append(wrap_chunk(current, bos=bos, eos=eos))
        current = []
        last_comma = -1

    for token in tokens:
        if comma_id is not None and token == comma_id:
            last_comma = len(current)
        elif (
            backtrack
            and len(current) == CLIP_CONTENT_LENGTH
            and last_comma != -1
            and len(current) - last_comma <= backtrack
        ):
            break_at = last_comma + 1
            reloc = current[break_at:]
            current = current[:break_at]
            flush()
            current = reloc

        if len(current) == CLIP_CONTENT_LENGTH:
            flush()
        current.append(token)

    if current or not chunks:
        flush()
    return chunks


def build_prompt_chunks(text: str, tokenizer: Any) -> list[list[int]]:
    bos, eos = _special_token_ids(tokenizer)
    comma_id = comma_token_id(tokenizer)
    segments = BREAK_RE.split(text) if text else [""]
    chunks: list[list[int]] = []
    for segment in segments:
        tokens = tokenize_without_specials(tokenizer, segment) if segment else []
        chunks.extend(pack_content_tokens(tokens, bos=bos, eos=eos, comma_id=comma_id))
    return chunks


def count_tokenizer_chunks(text: str, tokenizer: Any) -> PromptChunkCounts:
    segments = BREAK_RE.split(text) if text else [""]
    token_count = 0
    chunk_count = 0
    bos, eos = _special_token_ids(tokenizer)
    comma_id = comma_token_id(tokenizer)
    for segment in segments:
        tokens = tokenize_without_specials(tokenizer, segment) if segment else []
        token_count += len(tokens)
        chunk_count += len(pack_content_tokens(tokens, bos=bos, eos=eos, comma_id=comma_id))
    return PromptChunkCounts(tokens=token_count, chunks=max(chunk_count, 1))


def count_prompt_chunks(text: str, tokenizer_1: Any, tokenizer_2: Any) -> PromptChunkCounts:
    counts_1 = count_tokenizer_chunks(text, tokenizer_1)
    counts_2 = count_tokenizer_chunks(text, tokenizer_2)
    return PromptChunkCounts(
        tokens=max(counts_1.tokens, counts_2.tokens),
        chunks=max(counts_1.chunks, counts_2.chunks),
    )


def format_prompt_chunk_summary(text: str, tokenizer_1: Any, tokenizer_2: Any) -> str:
    counts = count_prompt_chunks(text, tokenizer_1, tokenizer_2)
    label = "chunk" if counts.chunks == 1 else "chunks"
    return f"{counts.tokens} tokens, {counts.chunks} {label}"


def align_chunk_lists(chunks_1: list[list[int]], chunks_2: list[list[int]], tokenizer_1: Any, tokenizer_2: Any) -> tuple[list[list[int]], list[list[int]]]:
    target = max(len(chunks_1), len(chunks_2), 1)
    padded_1 = list(chunks_1)
    padded_2 = list(chunks_2)
    empty_1 = empty_chunk_ids(tokenizer_1)
    empty_2 = empty_chunk_ids(tokenizer_2)
    while len(padded_1) < target:
        padded_1.append(empty_1)
    while len(padded_2) < target:
        padded_2.append(empty_2)
    return padded_1, padded_2


def pad_embeds_to_length(embeds: Tensor, target_seq: int, empty_chunk: Tensor) -> Tensor:
    padded = embeds
    while padded.shape[1] < target_seq:
        chunk = empty_chunk.to(device=padded.device, dtype=padded.dtype)
        padded = torch.cat([padded, chunk], dim=1)
    return padded


def align_embed_pair(
    prompt_embeds: Tensor,
    negative_prompt_embeds: Tensor,
    empty_chunk_embeds: Tensor,
) -> tuple[Tensor, Tensor]:
    target = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
    return (
        pad_embeds_to_length(prompt_embeds, target, empty_chunk_embeds),
        pad_embeds_to_length(negative_prompt_embeds, target, empty_chunk_embeds),
    )
