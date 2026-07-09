"use client";

import useSWR from "swr";
import Link from "next/link";
import { useState } from "react";
import { PlusCircle, Trash2, Image as ImageIcon } from "lucide-react";
import PathInput from "@/components/PathInput";
import { datasetsApi } from "@/lib/api/datasets";
import type { Dataset } from "@/types";

function CreateDatasetModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [imageDir, setImageDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !imageDir) {
      setError("Name and image directory are required");
      return;
    }
    setSaving(true);
    try {
      await datasetsApi.create({ name, image_dir: imageDir });
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
      <div className="bg-surface border border-border rounded-2xl p-6 w-full max-w-md space-y-4">
        <h2 className="text-lg font-semibold text-text">Add Dataset</h2>
        {error && <div className="text-error text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs text-text-muted mb-1">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-dataset"
              className="w-full rounded-lg bg-bg border border-border px-3 py-2 text-text text-sm placeholder:text-text-muted focus:outline-none focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
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
              className="flex-1 border border-border rounded-lg py-2 text-text-muted hover:text-text text-sm"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="flex-1 bg-accent hover:bg-accent-hover text-bg rounded-lg py-2 text-sm font-medium disabled:opacity-50"
            >
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
    <Link
      href={`/datasets/${dataset.id}`}
      className="block bg-surface rounded-xl border border-border p-5 space-y-3 hover:border-accent transition-colors"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="font-semibold text-text">{dataset.name}</div>
          <div className="text-xs text-text-muted mt-0.5 break-all">{dataset.image_dir}</div>
          <div className="flex flex-wrap gap-2 mt-2 text-[10px]">
            {dataset.target_resolution != null && (
              <span className="px-1.5 py-0.5 rounded border border-border text-text-muted">
                {dataset.target_resolution}px
              </span>
            )}
            <span
              className={`px-1.5 py-0.5 rounded border ${
                dataset.preprocess_ready
                  ? "border-success/30 text-success"
                  : "border-warning/30 text-warning"
              }`}
            >
              {dataset.preprocess_ready ? "Ready" : "Not prepared"}
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={(event) => {
            event.preventDefault();
            onDelete();
          }}
          className="p-1.5 rounded hover:bg-white/10 text-error hover:text-error shrink-0"
        >
          <Trash2 size={14} />
        </button>
      </div>
      <div className="flex items-center gap-2 text-sm text-text-muted">
        <ImageIcon size={14} />
        <span>{images?.images.length ?? "…"} images</span>
      </div>
    </Link>
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
          <h1 className="text-2xl font-bold text-text">Datasets</h1>
          <p className="text-text-muted mt-1">Manage your training image datasets and tags</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-bg rounded-lg px-4 py-2 text-sm font-medium"
        >
          <PlusCircle size={15} />
          Add Dataset
        </button>
      </div>

      {isLoading ? (
        <div className="text-text-muted">Loading…</div>
      ) : !datasets?.length ? (
        <div className="text-center py-20 text-text-muted">
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
