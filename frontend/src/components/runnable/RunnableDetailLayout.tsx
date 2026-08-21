"use client";

import { Loader2 } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import BackLink from "@/components/ui/BackLink";
import type { RunnableStatus } from "@/types";

export interface RunnableDetailLayoutProps {
  backHref: string;
  backAriaLabel: string;
  title: string;
  subtitle?: string;
  status: RunnableStatus;
  isLoading: boolean;
  notFoundMessage: string;
  actions?: React.ReactNode;
  banner?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  backIconSize?: number;
}

export default function RunnableDetailLayout({
  backHref,
  backAriaLabel,
  title,
  subtitle,
  status,
  isLoading,
  notFoundMessage,
  actions,
  banner,
  children,
  className = "space-y-6",
  backIconSize,
}: RunnableDetailLayoutProps) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <BackLink href={backHref} aria-label={backAriaLabel} iconSize={backIconSize} />
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-text font-display truncate">{title}</h1>
            <div className="flex items-center gap-3 mt-1 flex-wrap">
              <StatusBadge status={status} />
              {subtitle && <p className="text-xs text-muted truncate">{subtitle}</p>}
            </div>
          </div>
        </div>
        {actions && <div className="flex items-center gap-2 flex-wrap shrink-0">{actions}</div>}
      </div>

      {banner}
      {children}
    </div>
  );
}

export function RunnableDetailNotFound({ message }: { message: string }) {
  return <div className="text-error py-20">{message}</div>;
}
