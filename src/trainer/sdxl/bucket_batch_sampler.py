"""Aspect-ratio batch sampler for SDXL LoRA training."""

import random
from collections.abc import Iterator, Sized

from torch.utils.data import BatchSampler


class BucketBatchSampler(BatchSampler):
    """Yield batches of indices grouped by bucket resolution."""

    def __init__(
        self,
        bucket_keys: list[str],
        batch_size: int,
        *,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._bucket_keys = bucket_keys
        self._buckets: dict[str, list[int]] = {}
        for index, key in enumerate(bucket_keys):
            self._buckets.setdefault(key, []).append(index)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for indices in self._buckets.values():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), self._batch_size):
                batch = shuffled[start : start + self._batch_size]
                if len(batch) < self._batch_size and self._drop_last:
                    continue
                batches.append(batch)
        random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self._buckets.values():
            count = len(indices)
            if self._drop_last:
                total += count // self._batch_size
            else:
                total += (count + self._batch_size - 1) // self._batch_size
        return total


def build_bucket_batch_sampler(
    dataset: Sized,
    bucket_keys: list[str],
    batch_size: int,
) -> BucketBatchSampler:
    if len(bucket_keys) != len(dataset):
        raise ValueError("bucket_keys length must match dataset length")
    return BucketBatchSampler(bucket_keys, batch_size)
