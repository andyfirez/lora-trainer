"use client";

import { useEffect, useState } from "react";
import PathInput from "@/components/PathInput";
import { datasetsApi } from "@/lib/api/datasets";

interface Props {
  open: boolean;
  dataset: { id: number; name: string; image_dir: string };
  onClose: () => void;
  onSaved: () => void;
}

export default function EditDatasetModal({ open, dataset, onClose, onSaved }: Props) {
  const [name, setName] = useState(dataset.name);
  const [imageDir, setImageDir] = useState(dataset.image_dir);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setName(dataset.name);
      setImageDir(dataset.image_dir);
      setError(null);
    }
  }, [open, dataset.name, dataset.image_dir]);

  if (!open) return null;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name || !imageDir) {
      setError("Name and image directory are required");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await datasetsApi.update(dataset.id, { name, image_dir: imageDir });
      onSaved();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error updating dataset");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-6 w-full max-w-md space-y-4">
        <h2 className="text-lg font-semibold text-white">Edit Dataset</h2>
        {error && <div className="text-red-400 text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">Name</label>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="my-dataset"
              className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-white text-sm placeholder-[var(--muted)] focus:outline-none focus:border-[var(--accent)]"
            />
          </div>
          <PathInput
            label="Image Directory"
            value={imageDir}
            onChange={setImageDir}
            placeholder="/path/to/images"
            pickerTitle="Select Image Directory"
            kind="directory"
          />
          <div className="flex gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 border border-[var(--border)] rounded-lg py-2 text-[var(--muted)] hover:text-white text-sm"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="flex-1 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg py-2 text-sm font-medium disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save Changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
