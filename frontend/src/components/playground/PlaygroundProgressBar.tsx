"use client";

interface PlaygroundProgressBarProps {
  label: string | null;
  pct: number | null;
}

export default function PlaygroundProgressBar({ label, pct }: PlaygroundProgressBarProps) {
  return (
    <div className="px-3 py-2 space-y-1">
      <div className="min-h-4 truncate text-xs text-muted">{label ?? "\u00a0"}</div>
      <div className="flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
          {pct != null ? (
            <div className="h-1.5 rounded-full bg-sampling transition-all" style={{ width: `${pct}%` }} />
          ) : label ? (
            <div className="progress-indeterminate h-1.5 rounded-full bg-sampling" />
          ) : null}
        </div>
        <span className="w-8 text-right text-xs text-muted">{pct != null ? `${pct}%` : "\u00a0"}</span>
      </div>
    </div>
  );
}
