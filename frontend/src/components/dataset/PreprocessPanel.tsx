"use client";

import { useState } from "react";
import { datasetsApi } from "@/lib/api/datasets";
import type { Dataset, PreprocessStatus } from "@/types";

interface Props {
  dataset: Dataset;
  status: PreprocessStatus | undefined;
  onUpdated: () => void;
}

export default function PreprocessPanel({ dataset, status, onUpdated }: Props) {
  const [resolution, setResolution] = useState(dataset.target_resolution ?? 1024);
  const [savingResolution, setSavingResolution] = useState(false);
  const [baking, setBaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSaveResolution = async () => {
    setSavingResolution(true);
    setError(null);
    try {
      await datasetsApi.update(dataset.id, { target_resolution: resolution });
      onUpdated();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save resolution");
    } finally {
      setSavingResolution(false);
    }
  };

  const handleBakeAll = async () => {
    setBaking(true);
    setError(null);
    try {
      await datasetsApi.bakeAll(dataset.id);
      onUpdated();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Bake failed");
    } finally {
      setBaking(false);
    }
  };

  const ready = status?.preprocess_ready ?? dataset.preprocess_ready;

  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 space-y-3">
      <div className="text-sm font-medium text-white">Preprocessing</div>
      <p className="text-xs text-[var(--muted)]">
        Set target resolution, then crop each image. Prepared images are saved for training at exactly this size.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-[var(--muted)] mb-1">Target resolution</label>
          <input
            type="number"
            min={64}
            max={2048}
            step={64}
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
            className="w-32 rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-1.5 text-sm text-white"
          />
        </div>
        <button
          type="button"
          onClick={handleSaveResolution}
          disabled={savingResolution}
          className="px-3 py-1.5 text-sm border border-[var(--border)] rounded-lg hover:bg-white/5 disabled:opacity-50"
        >
          {savingResolution ? "Saving…" : "Apply resolution"}
        </button>
        <button
          type="button"
          onClick={handleBakeAll}
          disabled={baking || !dataset.target_resolution}
          className="px-3 py-1.5 text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg disabled:opacity-50"
        >
          {baking ? "Baking…" : "Bake all"}
        </button>
      </div>

      {status && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span
            className={`px-2 py-0.5 rounded-full border ${
              ready
                ? "border-green-500/40 text-green-300 bg-green-500/10"
                : "border-amber-500/40 text-amber-300 bg-amber-500/10"
            }`}
          >
            {ready ? "Ready for training" : "Not ready"}
          </span>
          <span className="text-[var(--muted)]">{status.ready}/{status.total} baked</span>
          {status.no_crop > 0 && <span className="text-[var(--muted)]">{status.no_crop} need crop</span>}
          {status.stale > 0 && <span className="text-amber-400">{status.stale} stale</span>}
          {status.cropped > 0 && <span className="text-[var(--muted)]">{status.cropped} need bake</span>}
        </div>
      )}

      {error && <div className="text-sm text-red-400">{error}</div>}
    </div>
  );
}
