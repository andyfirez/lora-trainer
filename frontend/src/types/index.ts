export type RunnableStatus =
  | "draft"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "orphan";
export type TaggingMode = "if_empty" | "overwrite" | "append";

export interface RunnableResponse {
  id: number;
  name: string;
  status: RunnableStatus;
  queue_position: number | null;
  error_message: string | null;
  output_path: string | null;
  log_path: string | null;
  running_started_at: string | null;
  accumulated_elapsed_seconds: number;
  elapsed_seconds: number | null;
  created_at: string;
  updated_at: string;
}

export interface LoraResponse extends RunnableResponse {
  config_yaml: string | null;
  relative_path: string;
  weights_relpath: string;
  base_model_name: string;
  resolved_work_dir: string;
  resolved_weights_path: string;
  path_missing: boolean;
  can_resume: boolean;

  progress_step: number | null;
  progress_total: number | null;
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
}

export interface SamplingResponse extends RunnableResponse {
  config_yaml: string;
  lora_paths: string[];
  progress_step: number | null;
  progress_total: number | null;
  progress_status: string | null;
}

export interface RunnableSample {
  filename: string;
  path: string;
  url: string;
  kind?: "cell" | "grid" | "legacy";
  metadata?: Record<string, unknown>;
}

export interface RunnableSamplesResponse {
  samples: RunnableSample[];
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
  sampling_id: number | null;
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
}

export interface AutotagStatusResponse {
  status: "idle" | "running" | "completed" | "failed";
  current: number;
  total: number;
  message: string;
  error: string | null;
}

export interface PngInfoResponse {
  info: string;
  items: Record<string, string>;
  parameters: Record<string, string | number>;
  width: number;
  height: number;
  preview_base64: string | null;
}
