import type { RunnableStatus } from "@/types";

export interface QueueEntry {
  kind: "lora" | "sampling";
  id: number;
  label: string;
  status: RunnableStatus;
}
