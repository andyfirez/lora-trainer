"use client";

interface PlaygroundProgressBarProps {
  label: string | null;
  pct: number | null;
}

export default function PlaygroundProgressBar({ label, pct }: PlaygroundProgressBarProps) {
  if (!label && pct == null) return null;

  return (
    <div className="px-3 py-2 space-y-1">
      {label ? <div className="truncate text-xs text-muted">{label}</div> : null}
      {pct != null ? (
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 rounded-full bg-border">
            <div className="h-1.5 rounded-full bg-sampling transition-all" style={{ width: `${pct}%` }} />
          </div>
          <span className="text-xs text-muted">{pct}%</span>
        </div>
      ) : null}
    </div>
  );
}
