export type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
export type ConfigType = "training" | "sampling";
export type JobType = "training" | "sampling" | "tagging";
export type TaggingMode = "if_empty" | "overwrite" | "append";

export interface TrainingJobDetails {
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
}

export interface SamplingJobDetails {
  lora_paths: string[];
  source_job_id: number | null;
  progress_status: string | null;
}

export interface TaggingJobDetails {
  progress_status: string | null;
  dataset_id: number;
}

export interface Job {
  id: number;
  job_type: JobType;
  name: string;
  status: JobStatus;
  config_id: number | null;
  config_yaml: string;
  output_path: string | null;
  log_path: string | null;
  pid: number | null;
  error_message: string | null;
  progress_step: number | null;
  progress_total: number | null;
  training: TrainingJobDetails | null;
  sampling: SamplingJobDetails | null;
  tagging: TaggingJobDetails | null;
  can_resume: boolean;
  created_at: string;
  updated_at: string;
}

export interface JobConfig {
  id: number;
  name: string;
  config_type: ConfigType;
  config_yaml: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface QueueEntry {
  id: number;
  job_id: number;
  position: number;
  added_at: string;
}

export interface QueueEntryWithJob {
  entry: QueueEntry;
  job: Job;
}

export interface JobSample {
  filename: string;
  path: string;
  url: string;
}

export interface JobSamplesResponse {
  samples: JobSample[];
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

export interface DatasetItem {
  filename: string;
  tags: string[];
  has_caption: boolean;
}

export interface DatasetItems {
  dataset_id: number;
  items: DatasetItem[];
}

export interface TagStat {
  tag: string;
  count: number;
}

export interface TagStats {
  tags: TagStat[];
}

export interface BulkTagResult {
  updated_count: number;
}

export interface AutotagRequest {
  mode?: TaggingMode;
  threshold?: number;
  model?: string;
  caption_extension?: string;
  strip_rating?: boolean;
  filenames?: string[];
  enqueue?: boolean;
}

export interface AutotagResponse {
  job_id: number;
}

export interface CreateJobFromConfigRequest {
  name?: string;
  lora_paths?: string[];
  source_job_id?: number;
  enqueue?: boolean;
}
