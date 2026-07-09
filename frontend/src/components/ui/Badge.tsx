import { cn } from "@/lib/cn";
import type { HTMLAttributes } from "react";

const variants = {
  default: "bg-surface-raised text-text-secondary",
  accent: "bg-accent-muted text-accent",
  sampling: "bg-sampling-muted text-sampling",
  success: "bg-success-muted text-success",
  warning: "bg-warning-muted text-warning",
  error: "bg-error-muted text-error",
  running: "bg-running-muted text-running",
} as const;

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: keyof typeof variants;
}

export default function Badge({ className, variant = "default", children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </span>
  );
}
