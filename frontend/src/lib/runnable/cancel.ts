import type { LoraResponse, RunnableResponse } from "@/types";

export function canSaveCheckpointOnStop(runnable: RunnableResponse): boolean {
  if (!("last_checkpoint_path" in runnable)) return false;
  const lora = runnable as LoraResponse;
  return lora.progress_step != null && lora.progress_step > 0;
}

export function needsStopDialog(runnable: RunnableResponse): boolean {
  return runnable.status === "running";
}
