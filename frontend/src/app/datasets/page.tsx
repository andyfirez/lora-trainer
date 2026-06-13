"use client";

import useSWR from "swr";
import { useState } from "react";
import { PlusCircle, Trash2, Image as ImageIcon } from "lucide-react";
import PathInput from "@/components/PathInput";
import { datasetsApi } from "@/lib/api/datasets";
import type { Dataset } from "@/types";

function CreateDatasetModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [imageDir, setImageDir] = useState("");
  const [captionDir, setCaptionDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !imageDir) { setError("Name and image directory are required"); return; }
    setSaving(true);
    try {
      await datasetsApi.create({ name, image_dir: imageDir, caption_dir: captionDir || undefined });
      onCreated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error creating dataset");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-6 w-full max-w-md space-y-4">
        <h2 className="text-lg font-semibold text-white">Add Dataset</h2>
        {error && <div className="text-red-400 text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
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
          <PathInput
            label="Caption Directory (optional)"
            value={captionDir}
            onChange={setCaptionDir}
            placeholder="Same as image dir"
            pickerTitle="Select Caption Directory"
            kind="directory"
          />
          <div className="flex gap-2 pt-2">
            <button type="button" onClick={onClose} className="flex-1 border border-[var(--border)] rounded-lg py-2 text-[var(--muted)] hover:text-white text-sm">Cancel</button>
            <button type="submit" disabled={saving} className="flex-1 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg py-2 text-sm font-medium disabled:opacity-50">
              {saving ? "Adding…" : "Add Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function DatasetCard({ dataset, onDelete }: { dataset: Dataset; onDelete: () => void }) {
  const { data: images } = useSWR(`/datasets/${dataset.id}/images`, () => datasetsApi.listImages(dataset.id));

  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="font-semibold text-white">{dataset.name}</div>
          <div className="text-xs text-[var(--muted)] mt-0.5 break-all">{dataset.image_dir}</div>
        </div>
        <button onClick={onDelete} className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300 shrink-0">
          <Trash2 size={14} />
        </button>
      </div>
      <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
        <ImageIcon size={14} />
        <span>{images?.images.length ?? "…"} images</span>
      </div>
      {dataset.caption_dir && (
        <div className="text-xs text-[var(--muted)]">Captions: {dataset.caption_dir}</div>
      )}
    </div>
  );
}

export default function DatasetsPage() {
  const { data: datasets, isLoading, mutate } = useSWR("/datasets", () => datasetsApi.list());
  const [showCreate, setShowCreate] = useState(false);

  const handleDelete = async (id: number, name: string) => {
    if (!confirm(`Delete dataset "${name}"?`)) return;
    await datasetsApi.delete(id);
    mutate();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Datasets</h1>
          <p className="text-[var(--muted)] mt-1">Manage your training image datasets</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium"
        >
          <PlusCircle size={15} />
          Add Dataset
        </button>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading…</div>
      ) : !datasets?.length ? (
        <div className="text-center py-20 text-[var(--muted)]">
          No datasets yet. Add one to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {datasets.map((d) => (
            <DatasetCard key={d.id} dataset={d} onDelete={() => handleDelete(d.id, d.name)} />
          ))}
        </div>
      )}

      {showCreate && (
        <CreateDatasetModal onClose={() => setShowCreate(false)} onCreated={() => mutate()} />
      )}
    </div>
  );
}
