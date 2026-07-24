"""Tests for DatasetResponse path population."""

from datetime import datetime, timezone

from src.api.schemas.datasets import DatasetResponse
from src.db.tables.dataset import Dataset
from src.settings.app_settings import settings


def test_dataset_response_from_orm_object(storage_roots) -> None:
    dataset = Dataset(
        id=1,
        name="demo",
        relative_path="images",
        description=None,
        target_resolution=1024,
        preprocess_ready=False,
        enable_bucket=False,
        bucket_reso_steps=64,
        min_bucket_reso=512,
        max_bucket_reso=2048,
        bucket_no_upscale=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    (storage_roots["datasets"] / "images").mkdir()

    response = DatasetResponse.model_validate(dataset)
    assert response.relative_path == "images"
    assert response.resolved_path == str((storage_roots["datasets"] / "images").resolve())
    assert response.path_missing is False
