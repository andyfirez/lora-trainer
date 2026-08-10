import type { LoraResponse } from "@/types";

export function canSaveCheckpointOnStop(lora: LoraResponse): boolean {
  return lora.progress_step != null && lora.progress_step > 0;
}

export function needsStopDialog(lora: LoraResponse): boolean {
  return lora.status === "running";
}
