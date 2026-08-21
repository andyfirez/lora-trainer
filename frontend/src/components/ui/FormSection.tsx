import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface FormSectionProps {
  title: ReactNode;
  children: ReactNode;
  className?: string;
}

export default function FormSection({ title, children, className }: FormSectionProps) {
  return (
    <section className={cn("bg-surface rounded-xl border border-border p-5 space-y-4", className)}>
      <div className="text-sm font-semibold text-text mb-3 font-display">{title}</div>
      {children}
    </section>
  );
}

export const formSectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
export const formSectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";
