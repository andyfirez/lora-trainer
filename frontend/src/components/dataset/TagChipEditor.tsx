"use client";

import { KeyboardEvent, useState } from "react";
import { Plus } from "lucide-react";
import TagChip from "@/components/dataset/TagChip";

interface Props {
  tags: string[];
  onChange: (tags: string[]) => void;
  disabled?: boolean;
}

export default function TagChipEditor({ tags, onChange, disabled = false }: Props) {
  const [input, setInput] = useState("");

  const addTag = (raw: string) => {
    const tag = raw.trim();
    if (!tag || tags.includes(tag)) {
      setInput("");
      return;
    }
    onChange([...tags, tag]);
    setInput("");
  };

  const removeTag = (tag: string) => {
    onChange(tags.filter((item) => item !== tag));
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addTag(input);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5 min-h-[1.75rem]">
        {tags.map((tag) => (
          <TagChip
            key={tag}
            label={tag}
            onRemove={disabled ? undefined : () => removeTag(tag)}
            disabled={disabled}
          />
        ))}
      </div>
      {!disabled && (
        <div className="flex gap-1">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => addTag(input)}
            placeholder="Add tag…"
            className="flex-1 rounded-md bg-bg border border-border px-2 py-1 text-xs text-text placeholder-muted focus:outline-none focus:border-accent"
          />
          <button
            type="button"
            onClick={() => addTag(input)}
            className="rounded-md border border-border px-2 text-muted hover:text-text"
          >
            <Plus size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
