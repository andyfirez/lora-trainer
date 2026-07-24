"""Tests that training config rejects unprepared datasets."""

import pytest
from src.db.tables.job_config import ConfigType
from src.services.configs.exceptions import JobConfigValidationError
from src.services.configs.service import JobConfigService
from src.services.datasets.service import DatasetsService


@pytest.mark.asyncio
async def test_create_config_rejects_unprepared_dataset(
    config_service: JobConfigService,
    datasets_service: DatasetsService,
    storage_roots,
) -> None:
    from PIL import Image

    image_dir = storage_roots["datasets"] / "images"
    image_dir.mkdir()
    Image.new("RGB", (1024, 1024)).save(image_dir / "test.png")
    dataset = await datasets_service.create_dataset(name="raw", relative_path="images")
    await datasets_service.update_dataset(
        dataset.id,
        name=None,
        relative_path=None,
        description=None,
        target_resolution=1024,
        update_target_resolution=True,
    )

    yaml_text = f"""base_model_name: test-model
resolution: 1024
concepts:
  - dataset_id: {dataset.id}
"""
    with pytest.raises(JobConfigValidationError, match="not ready for training"):
        await config_service.create_config(
            name="bad-config",
            config_type=ConfigType.TRAINING,
            config_yaml=yaml_text,
        )
