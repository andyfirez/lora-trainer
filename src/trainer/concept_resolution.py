"""Resolve training concept dataset IDs to image directories."""

from src.db.repositories.dataset_repo import DatasetRepository
from src.services.datasets.exceptions import DatasetNotFoundError


async def resolve_dataset_ids(dataset_ids: list[int], repo: DatasetRepository) -> dict[int, str]:
    """Load image_dir for each unique dataset id."""
    result: dict[int, str] = {}
    unique_ids = list(dict.fromkeys(dataset_ids))
    for dataset_id in unique_ids:
        dataset = await repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        result[dataset_id] = dataset.image_dir
    return result
