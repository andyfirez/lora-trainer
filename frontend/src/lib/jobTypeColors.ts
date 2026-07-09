import type { JobType } from "@/types";
import { cn } from "@/lib/cn";

export function jobLinkClass(jobType: JobType): string {
  switch (jobType) {
    case "sampling":
      return "hover:text-sampling";
    case "tagging":
      return "hover:text-tagging";
    default:
      return "hover:text-accent";
  }
}

export function jobBarClass(jobType: JobType): string {
  switch (jobType) {
    case "sampling":
      return "bg-sampling";
    case "tagging":
      return "bg-tagging";
    default:
      return "bg-accent";
  }
}

export function jobBadgeVariant(jobType: JobType): "accent" | "sampling" | "tagging" {
  switch (jobType) {
    case "sampling":
      return "sampling";
    case "tagging":
      return "tagging";
    default:
      return "accent";
  }
}

export function jobButtonClass(jobType: JobType, action: "primary" | "run" = "primary"): string {
  if (jobType === "sampling") {
    return action === "run"
      ? "bg-sampling/20 hover:bg-sampling/30 text-sampling border border-sampling/30"
      : "bg-sampling hover:bg-sampling/90 text-bg";
  }
  return action === "run"
    ? "bg-success/20 hover:bg-success/30 text-success border border-success/30"
    : "bg-accent hover:bg-accent-hover text-bg";
}

export function actionIconClass(action: "play" | "resume" | "queue" | "danger"): string {
  const base = "p-1.5 rounded transition-colors hover:bg-white/10";
  switch (action) {
    case "play":
      return cn(base, "text-success hover:text-success");
    case "resume":
      return cn(base, "text-running hover:text-running");
    case "queue":
      return cn(base, "text-warning hover:text-warning");
    case "danger":
      return cn(base, "text-error hover:text-error");
  }
}
