export type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
export type QueueItemType = "training" | "sampling";
export type SamplingRunStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: number;
  name: string;
  config_yaml: string;
  status: JobStatus;
  output_path: string | null;
  log_path: string | null;
  pid: number | null;
  error_message: string | null;
  progress_step: number | null;
  progress_total: number | null;
  progress_loss: number | null;
  progress_avr_loss: number | null;
  progress_epoch: number | null;
  progress_epoch_total: number | null;
  cache_progress_step: number | null;
  cache_progress_total: number | null;
  sampling_status: string | null;
  sampling_step: number | null;
  sampling_total: number | null;
  last_checkpoint_path: string | null;
  last_checkpoint_epoch: number | null;
  last_checkpoint_step: number | null;
  resume_checkpoint_path: string | null;
  resume_from_epoch: number | null;
  resume_from_step: number | null;
  save_checkpoint_requested: boolean;
  can_resume: boolean;
  created_at: string;
  updated_at: string;
}

export interface QueueEntry {
  id: number;
  item_type: QueueItemType;
  item_id: number;
  position: number;
  added_at: string;
}

export interface SamplingRun {
  id: number;
  name: string;
  config_yaml: string;
  lora_paths: string[];
  status: SamplingRunStatus;
  source_job_id: number | null;
  output_path: string | null;
  log_path: string | null;
  pid: number | null;
  error_message: string | null;
  progress_status: string | null;
  progress_step: number | null;
  progress_total: number | null;
  created_at: string;
  updated_at: string;
}

export interface SamplingRunSample {
  filename: string;
  path: string;
  url: string;
}

export interface SamplingRunSamplesResponse {
  samples: SamplingRunSample[];
}

export interface QueueEntryWithItem {
  entry: QueueEntry;
  job: Job | null;
  sampling_run: SamplingRun | null;
}

export interface Dataset {
  id: number;
  name: string;
  image_dir: string;
  caption_dir: string | null;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetImages {
  dataset_id: number;
  image_dir: string;
  images: string[];
}
