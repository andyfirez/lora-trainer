"""Resolve training concept dataset IDs to image directories."""

from dataclasses import dataclass

from src.db.repositories.dataset_repo import DatasetRepository
from src.services.datasets.exceptions import DatasetNotFoundError
from src.services.datasets.paths import dataset_image_dir_str
from src.services.datasets.preprocess import prepared_dir_path
from src.storage.paths import StoragePaths


@dataclass(frozen=True)
class ResolvedConceptPaths:
    image_dir: str
    prepared_dir: str


async def resolve_dataset_ids(dataset_ids: list[int], repo: DatasetRepository) -> dict[int, str]:
    """Load image_dir for each unique dataset id."""
    paths = await resolve_concept_paths(dataset_ids, repo)
    return {dataset_id: entry.image_dir for dataset_id, entry in paths.items()}


async def resolve_concept_paths(
    dataset_ids: list[int],
    repo: DatasetRepository,
) -> dict[int, ResolvedConceptPaths]:
    """Load image_dir and prepared_dir for each unique dataset id."""
    result: dict[int, ResolvedConceptPaths] = {}
    unique_ids = list(dict.fromkeys(dataset_ids))
    for dataset_id in unique_ids:
        dataset = await repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if dataset.target_resolution is None:
            raise ValueError(f"Dataset with id={dataset_id} has no target_resolution")
        image_dir = dataset_image_dir_str(dataset)
        prepared_dir = prepared_dir_path(image_dir, dataset.target_resolution)
        result[dataset_id] = ResolvedConceptPaths(
            image_dir=image_dir,
            prepared_dir=str(prepared_dir),
        )
    return result
