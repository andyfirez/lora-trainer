import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export type AlertVariant = "info" | "success" | "warning" | "error";

const variantClass: Record<AlertVariant, string> = {
  info: "border-accent/30 bg-accent-muted text-text",
  success: "border-success/30 bg-success-muted text-text",
  warning: "border-warning/30 bg-warning-muted text-text",
  error: "border-error/30 bg-error-muted text-error",
};

export interface AlertProps {
  variant?: AlertVariant;
  children: ReactNode;
  className?: string;
}

export default function Alert({ variant = "info", children, className }: AlertProps) {
  return (
    <div className={cn("rounded-xl border px-4 py-3 text-sm", variantClass[variant], className)}>
      {children}
    </div>
  );
}
