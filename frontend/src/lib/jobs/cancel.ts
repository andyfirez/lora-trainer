import type { Job } from "@/types";

export function canSaveCheckpointOnStop(job: Job): boolean {
  return job.job_type === "training" && job.progress_step != null && job.progress_step > 0;
}

export function needsStopDialog(job: Job): boolean {
  return job.status === "running" && job.job_type === "training";
}
