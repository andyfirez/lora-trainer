export type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";

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
