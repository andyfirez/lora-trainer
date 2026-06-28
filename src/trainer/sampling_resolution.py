"""Resolve sampling config IDs to SamplingConfig for training jobs."""

from src.db.repositories.job_config_repo import JobConfigRepository
from src.db.tables.job_config import ConfigType
from src.sampler.config import SamplingConfig


async def resolve_sampling_config(
    sampling_config_id: int | None,
    repo: JobConfigRepository,
) -> SamplingConfig | None:
    """Load SamplingConfig for the given job config id."""
    if sampling_config_id is None:
        return None
    entity = await repo.get_by_id(sampling_config_id)
    if entity is None:
        raise ValueError(f"Sampling config with id={sampling_config_id} not found")
    if entity.config_type != ConfigType.SAMPLING:
        raise ValueError(f"Job config id={sampling_config_id} is not a sampling config")
    return SamplingConfig.from_yaml(entity.config_yaml)
