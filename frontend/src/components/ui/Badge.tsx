import { cn } from "@/lib/cn";

type BadgeVariant =
  | "default"
  | "accent"
  | "success"
  | "warning"
  | "error"
  | "running"
  | "sampling"
  | "tagging"
  | "muted";

const variantClasses: Record<BadgeVariant, string> = {
  default: "bg-surface-raised text-text border border-border",
  accent: "bg-accent-muted text-accent",
  success: "bg-success-muted text-success",
  warning: "bg-warning-muted text-warning",
  error: "bg-error-muted text-error",
  running: "bg-running-muted text-running",
  sampling: "bg-sampling-muted text-sampling",
  tagging: "bg-tagging-muted text-tagging",
  muted: "bg-white/5 text-text-muted",
};

export interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

export default function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        variantClasses[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
