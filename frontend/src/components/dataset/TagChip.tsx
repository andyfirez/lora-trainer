"use client";

import { X } from "lucide-react";

interface Props {
  label: string;
  count?: number;
  onRemove?: () => void;
  disabled?: boolean;
}

export default function TagChip({ label, count, onRemove, disabled = false }: Props) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-white/10 px-2 py-0.5 text-xs text-white">
      <span>{label}</span>
      {count != null && (
        <span className="text-[var(--muted)] tabular-nums">{count}</span>
      )}
      {onRemove != null && (
        <button
          type="button"
          onClick={onRemove}
          disabled={disabled}
          className="text-[var(--muted)] hover:text-white disabled:opacity-40"
          aria-label={`Remove ${label}`}
        >
          <X size={12} />
        </button>
      )}
    </span>
  );
}
