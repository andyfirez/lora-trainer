"""API response converters."""


from src.api.schemas.jobs import (
    JobResponse,
    SamplingJobDetails,
    TaggingJobDetails,
    TrainingJobDetails,
)
from src.db.tables.job import Job, JobType
from src.services.jobs.service import JobsService
from src.tagger.config import TaggingConfig
from src.trainer.config import TrainConfig


def to_job_response(job: Job, service: JobsService) -> JobResponse:
    payload = {
        "id": job.id,
        "job_type": job.job_type,
        "name": job.name,
        "status": job.status,
        "config_id": job.config_id,
        "config_yaml": job.config_yaml,
        "output_path": job.output_path,
        "log_path": job.log_path,
        "pid": job.pid,
        "error_message": job.error_message,
        "progress_step": job.progress_step,
        "progress_total": job.progress_total,
        "can_resume": service.can_resume(job),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.job_type == JobType.TRAINING:
        train_config = TrainConfig.from_yaml(job.config_yaml)
        payload["training"] = TrainingJobDetails(
            progress_loss=job.progress_loss,
            progress_avr_loss=job.progress_avr_loss,
            progress_epoch=job.progress_epoch,
            progress_epoch_total=job.progress_epoch_total,
            cache_progress_step=job.cache_progress_step,
            cache_progress_total=job.cache_progress_total,
            sampling_status=job.sampling_status,
            sampling_step=job.sampling_step,
            sampling_total=job.sampling_total,
            last_checkpoint_path=job.last_checkpoint_path,
            last_checkpoint_epoch=job.last_checkpoint_epoch,
            last_checkpoint_step=job.last_checkpoint_step,
            resume_checkpoint_path=job.resume_checkpoint_path,
            resume_from_epoch=job.resume_from_epoch,
            resume_from_step=job.resume_from_step,
            save_checkpoint_requested=job.save_checkpoint_requested,
            sampling_config_id=train_config.sampling_config_id,
        )
    elif job.job_type == JobType.SAMPLING:
        payload["sampling"] = SamplingJobDetails(
            lora_paths=service.get_lora_paths(job),
            source_job_id=job.source_job_id,
            progress_status=job.progress_status,
        )
    elif job.job_type == JobType.TAGGING:
        payload["tagging"] = TaggingJobDetails(
            progress_status=job.progress_status,
            dataset_id=TaggingConfig.from_yaml(job.config_yaml).dataset_id,
        )
    return JobResponse.model_validate(payload)
