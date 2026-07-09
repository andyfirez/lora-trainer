"use client";

import { useEffect, useRef, useState } from "react";
import TagChipEditor from "@/components/dataset/TagChipEditor";
import { datasetImageUrl, datasetPreparedImageUrl } from "@/lib/api/datasets";
import type { ImagePreprocessState } from "@/types";

interface Props {
  datasetId: number;
  filename: string;
  initialTags: string[];
  preprocessState?: ImagePreprocessState | null;
  canCrop: boolean;
  preparing?: boolean;
  cacheKey?: string;
  onCropClick: () => void;
  onTagsSaved: (filename: string, tags: string[]) => void;
  onSave: (filename: string, tags: string[]) => Promise<void>;
}

const STATE_LABELS: Record<ImagePreprocessState, { label: string; className: string }> = {
  no_crop: { label: "Needs crop", className: "bg-warning-muted text-warning border-warning/30" },
  stale: { label: "Stale", className: "bg-accent-muted text-accent border-accent/30" },
  cropped: { label: "Needs bake", className: "bg-running-muted text-running border-running/30" },
  ready: { label: "Ready", className: "bg-success-muted text-success border-success/30" },
};

export default function DatasetImageCard({
  datasetId,
  filename,
  initialTags,
  preprocessState,
  canCrop,
  preparing = false,
  cacheKey,
  onCropClick,
  onTagsSaved,
  onSave,
}: Props) {
  const [tags, setTags] = useState(initialTags);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preparedFailed, setPreparedFailed] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestTags = useRef(tags);

  useEffect(() => {
    setTags(initialTags);
  }, [initialTags]);

  useEffect(() => {
    setPreparedFailed(false);
  }, [preprocessState, cacheKey, filename]);

  useEffect(() => {
    latestTags.current = tags;
  }, [tags]);

  const scheduleSave = (nextTags: string[]) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      setSaving(true);
      setError(null);
      try {
        await onSave(filename, nextTags);
        onTagsSaved(filename, nextTags);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Save failed");
      } finally {
        setSaving(false);
      }
    }, 500);
  };

  const handleChange = (nextTags: string[]) => {
    setTags(nextTags);
    scheduleSave(nextTags);
  };

  useEffect(() => {
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, []);

  const stateInfo = preprocessState ? STATE_LABELS[preprocessState] : null;
  const badgeLabel =
    preparing && preprocessState && preprocessState !== "ready"
      ? "Preparing…"
      : stateInfo?.label;
  const badgeClassName =
    preparing && preprocessState && preprocessState !== "ready"
      ? "bg-running-muted text-running border-running/30"
      : stateInfo?.className;
  const usePrepared = preprocessState === "ready" && !preparedFailed;
  const thumbnailSrc = usePrepared
    ? datasetPreparedImageUrl(datasetId, filename, 256, cacheKey)
    : datasetImageUrl(datasetId, filename, 256, cacheKey);

  return (
    <div className="bg-surface rounded-xl border border-border overflow-hidden flex flex-col">
      <button
        type="button"
        onClick={canCrop ? onCropClick : undefined}
        disabled={!canCrop}
        className={`aspect-square bg-bg relative w-full text-left ${canCrop ? "cursor-pointer hover:opacity-90" : "cursor-default"}`}
        title={canCrop ? "Edit crop" : "Set target resolution first"}
      >
        <img
          key={thumbnailSrc}
          src={thumbnailSrc}
          alt={filename}
          className="w-full h-full object-cover pointer-events-none"
          loading="lazy"
          onError={() => {
            if (usePrepared) setPreparedFailed(true);
          }}
        />
        {badgeLabel && badgeClassName && (
          <span
            className={`absolute top-2 left-2 text-[10px] px-1.5 py-0.5 rounded border ${badgeClassName}`}
          >
            {badgeLabel}
          </span>
        )}
      </button>
      <div className="p-3 space-y-2 flex-1">
        <div className="text-xs text-muted truncate" title={filename}>
          {filename}
        </div>
        <TagChipEditor tags={tags} onChange={handleChange} disabled={saving} />
        <div className="text-[10px] text-muted min-h-[14px]">
          {saving ? "Saving…" : error ? <span className="text-error">{error}</span> : null}
        </div>
      </div>
    </div>
  );
}
