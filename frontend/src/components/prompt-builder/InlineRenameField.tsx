"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";

interface InlineRenameFieldProps {
  value: string;
  onSave: (next: string) => Promise<void> | void;
  ariaLabel: string;
  className?: string;
  inputClassName?: string;
  title?: string;
}

export default function InlineRenameField({
  value,
  onSave,
  ariaLabel,
  className,
  inputClassName,
  title = "Click to rename",
}: InlineRenameFieldProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  async function commit() {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === value) {
      setDraft(value);
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      await onSave(trimmed);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  function cancel() {
    setDraft(value);
    setEditing(false);
  }

  if (editing) {
    return (
      <input
        autoFocus
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void commit();
          }
          if (event.key === "Escape") {
            event.preventDefault();
            cancel();
          }
        }}
        disabled={saving}
        aria-label={ariaLabel}
        className={cn(
          "min-w-0 rounded border border-accent/50 bg-input px-1.5 py-0.5 text-inherit outline-none focus:border-accent focus:ring-1 focus:ring-accent/30",
          inputClassName,
        )}
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => setEditing(true)}
      className={cn("min-w-0 truncate text-left hover:underline decoration-dotted underline-offset-2", className)}
      title={title}
    >
      {value}
    </button>
  );
}
