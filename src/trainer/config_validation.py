"""Shared TrainConfig validation for enqueue and handler checks."""

from src.db.repositories.dataset_repo import DatasetRepository
from src.services.datasets.training_validation import validate_dataset_for_training
from src.services.runnable.exceptions import RunnableValidationError
from src.storage.config_paths import resolve_config_base_model
from src.storage.paths import StoragePaths
from src.trainer.config import TrainConfig


class TrainConfigValidator:
    @staticmethod
    async def validate_for_enqueue(config: TrainConfig, dataset_repo: DatasetRepository) -> None:
        """Validate a training config before starting or resuming a LoRA run."""
        try:
            if not config.base_model_name:
                raise RunnableValidationError("base_model_name is required")
            resolve_config_base_model(config.base_model_name)
            StoragePaths.resolve_lora_path(config.output_dir or "")
            if not config.concepts:
                raise RunnableValidationError("At least one training concept is required")
            for concept in config.concepts:
                dataset = await dataset_repo.get_by_id(concept.dataset_id)
                if dataset is None:
                    raise RunnableValidationError(f"Dataset with id={concept.dataset_id} not found")
                validate_dataset_for_training(
                    dataset,
                    config.resolution,
                    enable_bucket=config.enable_bucket,
                )
            config.validate_gpu()
        except RunnableValidationError:
            raise
        except Exception as exc:
            raise RunnableValidationError(str(exc)) from exc
