import type { RunnableStatus } from "@/types";

const ENQUEUE_STATUSES: ReadonlySet<RunnableStatus> = new Set(["draft", "failed", "cancelled", "orphan"]);
const CANCEL_STATUSES: ReadonlySet<RunnableStatus> = new Set(["queued", "running"]);

export function canEnqueue(status: RunnableStatus): boolean {
  return ENQUEUE_STATUSES.has(status);
}

export function canCancel(status: RunnableStatus): boolean {
  return CANCEL_STATUSES.has(status);
}

export function cancelActionLabel(status: RunnableStatus): "Stop" | "Cancel" {
  return status === "running" ? "Stop" : "Cancel";
}
