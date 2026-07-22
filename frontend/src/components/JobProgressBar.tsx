"use client";

import type { ReactNode } from "react";
import { Loader2 } from "lucide-react";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";

interface Props {
  title: string;
  step: number | null;
  total: number | null;
  percent: number;
  active: boolean;
  elapsedSeconds?: number | null;
  barClassName?: string;
  titleExtra?: ReactNode;
  headerRight?: ReactNode;
  footer?: ReactNode;
  showSpinner?: boolean;
  showBar?: boolean;
}

export default function JobProgressBar({
  title,
  step,
  total,
  percent,
  active,
  elapsedSeconds = null,
  barClassName = "bg-accent",
  titleExtra,
  headerRight,
  footer,
  showSpinner = false,
  showBar = true,
}: Props) {
  return (
    <div className="bg-surface rounded-xl border border-border p-5 space-y-3">
      <div className="flex justify-between text-sm">
        <span className="text-text font-medium flex items-center gap-2">
          {showSpinner && <Loader2 className="animate-spin text-accent" size={14} />}
          {title}
          {titleExtra}
        </span>
        {headerRight}
      </div>
      {showBar && (
        <div className="bg-border rounded-full h-2.5">
          <div
            className={`${barClassName} h-2.5 rounded-full transition-all duration-500`}
            style={{ width: `${percent}%` }}
          />
        </div>
      )}
      <ProgressTimingInfo
        step={step}
        total={total}
        active={active}
        elapsedSeconds={elapsedSeconds}
      />
      {footer}
    </div>
  );
}
