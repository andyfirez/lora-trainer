"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { datasetsApi } from "@/lib/api/datasets";
import { useDebouncedCallback } from "@/hooks/useDebouncedCallback";
import type { Dataset, PreprocessStatus } from "@/types";

const DEFAULT_RESOLUTION = 1024;
const MIN_RESOLUTION = 64;
const MAX_RESOLUTION = 2048;

interface Props {
  dataset: Dataset;
  status: PreprocessStatus | undefined;
  preparing: boolean;
  onUpdated: () => void;
}

function clampResolution(value: number): number {
  return Math.min(MAX_RESOLUTION, Math.max(MIN_RESOLUTION, value));
}

export default function PreprocessPanel({ dataset, status, preparing, onUpdated }: Props) {
  const [resolution, setResolution] = useState(dataset.target_resolution ?? DEFAULT_RESOLUTION);
  const [savingResolution, setSavingResolution] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initStarted = useRef(false);

  useEffect(() => {
    initStarted.current = false;
  }, [dataset.id]);

  useEffect(() => {
    setResolution(dataset.target_resolution ?? DEFAULT_RESOLUTION);
  }, [dataset.target_resolution]);

  const saveResolution = useCallback(
    async (value: number) => {
      const clamped = clampResolution(value);
      if (clamped === dataset.target_resolution) return;
      setSavingResolution(true);
      setError(null);
      try {
        await datasetsApi.update(dataset.id, { target_resolution: clamped });
        onUpdated();
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to save resolution");
      } finally {
        setSavingResolution(false);
      }
    },
    [dataset.id, dataset.target_resolution, onUpdated]
  );

  const debouncedSaveResolution = useDebouncedCallback(
    (value: number) => {
      void saveResolution(value);
    },
    500
  );

  useEffect(() => {
    if (dataset.target_resolution != null || initStarted.current) return;
    initStarted.current = true;
    void saveResolution(DEFAULT_RESOLUTION);
  }, [dataset.target_resolution, saveResolution]);

  const handleResolutionChange = (value: number) => {
    const clamped = clampResolution(value);
    setResolution(clamped);
    debouncedSaveResolution(clamped);
  };

  const ready = status?.preprocess_ready ?? dataset.preprocess_ready;

  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 space-y-3">
      <div className="text-sm font-medium text-white">Preprocessing</div>
      <p className="text-xs text-[var(--muted)]">
        Images are prepared automatically. Click any image to adjust crop.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-[var(--muted)] mb-1">Target resolution</label>
          <input
            type="number"
            min={MIN_RESOLUTION}
            max={MAX_RESOLUTION}
            step={64}
            value={resolution}
            onChange={(e) => handleResolutionChange(Number(e.target.value))}
            className="w-32 rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-1.5 text-sm text-white"
          />
        </div>
        {(savingResolution || preparing) && (
          <span className="text-xs text-[var(--muted)] pb-1.5">
            {savingResolution ? "Saving resolution…" : "Preparing images…"}
          </span>
        )}
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
            {ready ? "Ready for training" : preparing ? "Preparing…" : "Not ready"}
          </span>
          <span className="text-[var(--muted)]">{status.ready}/{status.total} baked</span>
          {!preparing && status.no_crop > 0 && (
            <span className="text-[var(--muted)]">{status.no_crop} need crop</span>
          )}
          {!preparing && status.stale > 0 && (
            <span className="text-amber-400">{status.stale} stale</span>
          )}
          {!preparing && status.cropped > 0 && (
            <span className="text-[var(--muted)]">{status.cropped} need bake</span>
          )}
        </div>
      )}

      {error && <div className="text-sm text-red-400">{error}</div>}
    </div>
  );
}
