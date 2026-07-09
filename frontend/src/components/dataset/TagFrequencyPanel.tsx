"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import TagChip from "@/components/dataset/TagChip";
import type { TagStat } from "@/types";

interface Props {
  tags: TagStat[];
  onBulkAdd: (tag: string) => Promise<void>;
  onBulkRemove: (tag: string) => Promise<void>;
  disabled?: boolean;
}

export default function TagFrequencyPanel({ tags, onBulkAdd, onBulkRemove, disabled = false }: Props) {
  const [addInput, setAddInput] = useState("");
  const [busyTag, setBusyTag] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  const handleBulkAdd = async () => {
    const tag = addInput.trim();
    if (!tag) return;
    setAdding(true);
    try {
      await onBulkAdd(tag);
      setAddInput("");
    } finally {
      setAdding(false);
    }
  };

  const handleBulkRemove = async (tag: string) => {
    setBusyTag(tag);
    try {
      await onBulkRemove(tag);
    } finally {
      setBusyTag(null);
    }
  };

  return (
    <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted shrink-0">All tags</span>
        <div className="flex gap-1.5 min-w-[12rem] flex-1 max-w-xs">
          <input
            value={addInput}
            onChange={(event) => setAddInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void handleBulkAdd();
            }}
            placeholder="Add to all images…"
            disabled={disabled || adding}
            className="min-w-0 flex-1 rounded-md bg-bg border border-border px-2 py-1 text-xs text-text placeholder-muted focus:outline-none focus:border-accent"
          />
          <button
            type="button"
            onClick={() => void handleBulkAdd()}
            disabled={disabled || adding || !addInput.trim()}
            title="Add tag to all images"
            className="shrink-0 rounded-md border border-border px-2 py-1 text-muted hover:text-text disabled:opacity-50"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      {tags.length === 0 ? (
        <div className="text-xs text-muted">No tags yet</div>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {tags.map(({ tag, count }) => (
            <TagChip
              key={tag}
              label={tag}
              count={count}
              onRemove={() => void handleBulkRemove(tag)}
              disabled={disabled || busyTag === tag}
            />
          ))}
        </div>
      )}
    </div>
  );
}
