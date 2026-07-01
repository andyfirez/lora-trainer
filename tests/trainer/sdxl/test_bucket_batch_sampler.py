"""Tests for bucket batch sampler."""

from src.trainer.sdxl.bucket_batch_sampler import BucketBatchSampler


def test_bucket_batch_sampler_groups_by_resolution() -> None:
    bucket_keys = ["1024x768", "1024x768", "768x1024", "1024x768"]
    sampler = BucketBatchSampler(bucket_keys, batch_size=2)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        keys = {bucket_keys[index] for index in batch}
        assert len(keys) == 1
    assert sorted(sum(batches, [])) == [0, 1, 2, 3]
