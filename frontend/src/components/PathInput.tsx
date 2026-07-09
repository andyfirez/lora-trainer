"use client";

import { useState } from "react";
import { FolderOpen, Loader2 } from "lucide-react";
import FieldHint from "@/components/FieldHint";
import { inputClass } from "@/components/ui/Input";
import { filesApi, type PickKind } from "@/lib/api/files";

interface PathInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  pickerTitle: string;
  kind?: PickKind;
  placeholder?: string;
  hint?: string;
  hintAnchor?: string;
}

export default function PathInput({
  label,
  value,
  onChange,
  pickerTitle,
  kind = "file",
  placeholder,
  hint,
  hintAnchor,
}: PathInputProps) {
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleBrowse = async () => {
    setPicking(true);
    setError(null);
    try {
      const path = await filesApi.pick({
        kind,
        title: pickerTitle,
        initial_path: value || undefined,
      });
      if (path) onChange(path);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to open file dialog");
    } finally {
      setPicking(false);
    }
  };

  return (
    <div>
      <div className="flex items-center mb-1">
        <label className="block text-xs font-medium text-text-muted">{label}</label>
        {hint && <FieldHint hint={hint} hintAnchor={hintAnchor} />}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          className={inputClass}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
        />
        <button
          type="button"
          onClick={() => void handleBrowse()}
          disabled={picking}
          title="Browse"
          className="shrink-0 rounded-lg border border-border px-3 py-1.5 text-text-muted hover:bg-white/5 hover:text-text disabled:opacity-50"
        >
          {picking ? <Loader2 size={16} className="animate-spin" /> : <FolderOpen size={16} />}
        </button>
      </div>
      {error && <p className="mt-1 text-xs text-error">{error}</p>}
    </div>
  );
}
