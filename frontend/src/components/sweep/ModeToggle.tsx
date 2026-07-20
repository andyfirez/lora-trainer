"use client";

import type { SweepMode } from "@/lib/sweepUtils";

interface ModeToggleProps {
  mode: SweepMode;
  onChange: (mode: SweepMode) => void;
}

export default function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
    <div className="flex rounded-lg border border-border overflow-hidden text-xs shrink-0">
      <button
        type="button"
        onClick={() => onChange("fixed")}
        className={`px-3 py-1 ${mode === "fixed" ? "bg-accent text-white" : "text-muted hover:text-text"}`}
      >
        Fixed
      </button>
      <button
        type="button"
        onClick={() => onChange("vary")}
        className={`px-3 py-1 ${mode === "vary" ? "bg-accent text-white" : "text-muted hover:text-text"}`}
      >
        Vary
      </button>
    </div>
  );
}
