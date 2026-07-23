"use client";

import { useEffect, useState } from "react";
import { ChevronRight, FolderOpen, Loader2 } from "lucide-react";
import FieldHint from "@/components/FieldHint";
import Modal, { ModalFooter } from "@/components/ui/Modal";
import Button from "@/components/ui/Button";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { storageApi, type StorageKind } from "@/lib/api/storage";

interface StoragePathInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  kind: StorageKind;
  placeholder?: string;
  hint?: string;
  allowFiles?: boolean;
  warning?: string | null;
}

export default function StoragePathInput({
  label,
  value,
  onChange,
  kind,
  placeholder,
  hint,
  allowFiles = false,
  warning,
}: StoragePathInputProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<{ name: string; relative_path: string; is_dir: boolean }[]>([]);
  const [root, setRoot] = useState("");

  const loadEntries = async (relativePath: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await storageApi.browse(kind, relativePath);
      setRoot(response.root);
      setCurrentPath(response.relative_path);
      setEntries(response.entries);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to browse storage");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    void loadEntries(value || "");
  }, [open, kind, value]);

  const handleSelect = (relativePath: string) => {
    onChange(relativePath);
    setOpen(false);
  };

  return (
    <div className="w-full min-w-0">
      <div className="flex items-center mb-1">
        <label className={labelClassName}>{label}</label>
        {hint && <FieldHint hint={hint} />}
      </div>
      <div className="flex gap-2 min-w-0">
        <input
          type="text"
          className={`${inputClassName} flex-1 min-w-0`}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
        />
        <button
          type="button"
          onClick={() => setOpen(true)}
          title="Browse managed storage"
          className="shrink-0 rounded-lg border border-border px-3 py-1.5 text-muted hover:bg-white/5 hover:text-text"
        >
          <FolderOpen size={16} />
        </button>
      </div>
      {warning && <p className="mt-1 text-xs text-warning">{warning}</p>}
      {error && !open && <p className="mt-1 text-xs text-error">{error}</p>}

      <Modal open={open} title={`Browse ${kind.replace("_", " ")}`} onClose={() => setOpen(false)}>
        <div className="space-y-3">
          <p className="text-xs text-muted break-all">Root: {root || "…"}</p>
          <div className="flex items-center gap-1 text-sm text-muted">
            <button type="button" className="hover:text-text" onClick={() => void loadEntries("")}>
              /
            </button>
            {currentPath &&
              currentPath.split("/").map((part, index, parts) => {
                const path = parts.slice(0, index + 1).join("/");
                return (
                  <span key={path} className="flex items-center gap-1">
                    <ChevronRight size={12} />
                    <button type="button" className="hover:text-text" onClick={() => void loadEntries(path)}>
                      {part}
                    </button>
                  </span>
                );
              })}
          </div>
          {loading ? (
            <div className="flex items-center gap-2 text-muted py-6 justify-center">
              <Loader2 size={16} className="animate-spin" />
              Loading…
            </div>
          ) : (
            <div className="max-h-64 overflow-y-auto rounded-lg border border-border divide-y divide-border">
              <button
                type="button"
                className="w-full text-left px-3 py-2 hover:bg-white/5 text-sm"
                onClick={() => handleSelect(currentPath)}
              >
                Select current folder{currentPath ? `: ${currentPath}` : " (root)"}
              </button>
              {entries.map((entry) => (
                <div key={entry.relative_path} className="flex items-center justify-between px-3 py-2 hover:bg-white/5">
                  <button
                    type="button"
                    className="text-sm text-left flex-1"
                    onClick={() => (entry.is_dir ? void loadEntries(entry.relative_path) : handleSelect(entry.relative_path))}
                  >
                    {entry.is_dir ? "📁" : "📄"} {entry.name}
                  </button>
                  {!entry.is_dir && allowFiles && (
                    <Button size="sm" variant="secondary" onClick={() => handleSelect(entry.relative_path)}>
                      Select
                    </Button>
                  )}
                </div>
              ))}
              {!entries.length && <div className="px-3 py-4 text-sm text-muted">Empty folder</div>}
            </div>
          )}
          {error && open && <p className="text-sm text-error">{error}</p>}
        </div>
        <ModalFooter>
          <Button variant="secondary" onClick={() => setOpen(false)} className="flex-1">
            Close
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
