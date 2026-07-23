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
  last_checkpoint_path: string | null;
  last_checkpoint_epoch: number | null;
  last_checkpoint_step: number | null;
  resume_checkpoint_path: string | null;
  resume_from_epoch: number | null;
  resume_from_step: number | null;
  save_checkpoint_requested: boolean;
  sampling_config_id: number | null;
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
  elapsed_seconds: number | null;
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

export interface TrainedLora {
  id: number;
  name: string;
  job_id: number;
  config_id: number | null;
  config_yaml: string;
  base_model_name: string;
  weights_path: string;
  work_dir: string;
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
  kind?: "cell" | "grid" | "legacy";
  metadata?: Record<string, unknown>;
}

export interface JobSamplesResponse {
  samples: JobSample[];
}

export interface ManifestImageEntry {
  index: number;
  file: string;
  url: string;
  params: Record<string, unknown>;
  grid_position?: Record<string, number> | null;
}

export interface ManifestGridEntry {
  index: number;
  file: string;
  url: string;
  slice: Record<string, unknown>;
  x: { param: string; values: unknown[] };
  y: { param: string; values: unknown[] };
  cells: (number | null)[][];
  title: string;
}

export interface SweepManifestResponse {
  version: number;
  config_id: number | null;
  job_id: number | null;
  total_images: number;
  images: ManifestImageEntry[];
  grids: ManifestGridEntry[];
}

export interface Dataset {
  id: number;
  name: string;
  relative_path: string;
  resolved_path: string;
  path_missing: boolean;
  description: string | null;
  target_resolution: number | null;
  preprocess_ready: boolean;
  enable_bucket: boolean;
  bucket_reso_steps: number;
  min_bucket_reso: number;
  max_bucket_reso: number;
  bucket_no_upscale: boolean;
  created_at: string;
  updated_at: string;
}

export interface DatasetImages {
  dataset_id: number;
  relative_path: string;
  resolved_path: string;
  images: string[];
}

export interface DatasetItem {
  filename: string;
  tags: string[];
  has_caption: boolean;
  preprocess_state?: string | null;
}

export type ImagePreprocessState = "no_crop" | "stale" | "cropped" | "ready";

export interface PreprocessStatus {
  target_resolution: number | null;
  preprocess_ready: boolean;
  total: number;
  no_crop: number;
  stale: number;
  cropped: number;
  ready: number;
}

export interface DuplicatesInfo {
  duplicate_count: number;
}

export interface RemoveDuplicatesResult {
  removed_count: number;
}

export interface CropMeta {
  crop_center_x: number;
  crop_center_y: number;
  fitted_width: number;
  fitted_height: number;
  source_width: number;
  source_height: number;
  state: ImagePreprocessState;
  enable_bucket?: boolean;
  bucket_width?: number | null;
  bucket_height?: number | null;
  scale_to_width?: number | null;
  scale_to_height?: number | null;
  crop_x?: number;
  crop_y?: number;
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
