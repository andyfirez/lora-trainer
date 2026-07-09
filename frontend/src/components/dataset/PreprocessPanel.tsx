"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { datasetsApi } from "@/lib/api/datasets";
import { useDebouncedCallback } from "@/hooks/useDebouncedCallback";
import type { Dataset, PreprocessStatus } from "@/types";

const DEFAULT_RESOLUTION = 1024;
const MIN_RESOLUTION = 64;
const MAX_RESOLUTION = 2048;
const WINX_BUCKET_STEPS = 256;

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
  const [enableBucket, setEnableBucket] = useState(dataset.enable_bucket ?? false);
  const [bucketSteps, setBucketSteps] = useState(dataset.bucket_reso_steps ?? WINX_BUCKET_STEPS);
  const [bucketNoUpscale, setBucketNoUpscale] = useState(dataset.bucket_no_upscale ?? true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initStarted = useRef(false);

  useEffect(() => {
    initStarted.current = false;
  }, [dataset.id]);

  useEffect(() => {
    setResolution(dataset.target_resolution ?? DEFAULT_RESOLUTION);
    setEnableBucket(dataset.enable_bucket ?? false);
    setBucketSteps(dataset.bucket_reso_steps ?? WINX_BUCKET_STEPS);
    setBucketNoUpscale(dataset.bucket_no_upscale ?? true);
  }, [
    dataset.target_resolution,
    dataset.enable_bucket,
    dataset.bucket_reso_steps,
    dataset.bucket_no_upscale,
  ]);

  const saveSettings = useCallback(
    async (patch: Parameters<typeof datasetsApi.update>[1]) => {
      setSaving(true);
      setError(null);
      try {
        await datasetsApi.update(dataset.id, patch);
        onUpdated();
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to save settings");
      } finally {
        setSaving(false);
      }
    },
    [dataset.id, onUpdated]
  );

  const debouncedSaveResolution = useDebouncedCallback((value: number) => {
    void saveSettings({ target_resolution: clampResolution(value) });
  }, 500);

  useEffect(() => {
    if (dataset.target_resolution != null || initStarted.current) return;
    initStarted.current = true;
    void saveSettings({ target_resolution: DEFAULT_RESOLUTION });
  }, [dataset.target_resolution, saveSettings]);

  const handleResolutionChange = (value: number) => {
    const clamped = clampResolution(value);
    setResolution(clamped);
    debouncedSaveResolution(clamped);
  };

  const handleBucketToggle = async (checked: boolean) => {
    setEnableBucket(checked);
    await saveSettings({ enable_bucket: checked });
  };

  const handleBucketStepsChange = (value: number) => {
    setBucketSteps(value);
    void saveSettings({ bucket_reso_steps: value });
  };

  const handleBucketNoUpscaleToggle = async (checked: boolean) => {
    setBucketNoUpscale(checked);
    await saveSettings({ bucket_no_upscale: checked });
  };

  const ready = status?.preprocess_ready ?? dataset.preprocess_ready;

  return (
    <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
      <div className="text-sm font-medium text-text">Preprocessing</div>
      <p className="text-xs text-text-muted">
        {enableBucket
          ? "Aspect-ratio bucketing preserves image proportions. Click any image to adjust crop."
          : "Images are center-cropped to square. Click any image to adjust crop."}
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Target resolution</label>
          <input
            type="number"
            min={MIN_RESOLUTION}
            max={MAX_RESOLUTION}
            step={64}
            value={resolution}
            onChange={(e) => handleResolutionChange(Number(e.target.value))}
            className="w-32 rounded-lg bg-bg border border-border px-3 py-1.5 text-sm text-text"
          />
        </div>
        {(saving || preparing) && (
          <span className="text-xs text-text-muted pb-1.5">
            {saving ? "Saving…" : "Preparing images…"}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2 text-text cursor-pointer">
          <input
            type="checkbox"
            checked={enableBucket}
            onChange={(e) => void handleBucketToggle(e.target.checked)}
            className="rounded"
          />
          Enable bucketing
        </label>
        {enableBucket && (
          <>
            <div>
              <label className="block text-xs text-text-muted mb-1">Bucket steps</label>
              <input
                type="number"
                min={8}
                max={512}
                step={8}
                value={bucketSteps}
                onChange={(e) => handleBucketStepsChange(Number(e.target.value))}
                className="w-24 rounded-lg bg-bg border border-border px-2 py-1 text-sm text-text"
              />
            </div>
            <label className="flex items-center gap-2 text-text cursor-pointer">
              <input
                type="checkbox"
                checked={bucketNoUpscale}
                onChange={(e) => void handleBucketNoUpscaleToggle(e.target.checked)}
                className="rounded"
              />
              No upscale
            </label>
          </>
        )}
      </div>

      {status && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span
            className={`px-2 py-0.5 rounded-full border ${
              ready
                ? "border-success/40 text-success bg-success/10"
                : "border-warning/40 text-warning bg-warning/10"
            }`}
          >
            {ready ? "Ready for training" : preparing ? "Preparing…" : "Not ready"}
          </span>
          <span className="text-text-muted">{status.ready}/{status.total} baked</span>
          {!preparing && status.no_crop > 0 && (
            <span className="text-text-muted">{status.no_crop} need crop</span>
          )}
          {!preparing && status.stale > 0 && (
            <span className="text-warning">{status.stale} stale</span>
          )}
          {!preparing && status.cropped > 0 && (
            <span className="text-text-muted">{status.cropped} need bake</span>
          )}
        </div>
      )}

      {error && <div className="text-sm text-error">{error}</div>}
    </div>
  );
}
