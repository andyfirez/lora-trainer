import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface ErrorAlertProps {
  children: ReactNode;
  className?: string;
  compact?: boolean;
}

export default function ErrorAlert({ children, className, compact }: ErrorAlertProps) {
  return (
    <div
      className={cn(
        "rounded-lg bg-error-muted border border-error/30 text-error text-sm",
        compact ? "px-3 py-2" : "px-4 py-3",
        className,
      )}
    >
      {children}
    </div>
  );
}
