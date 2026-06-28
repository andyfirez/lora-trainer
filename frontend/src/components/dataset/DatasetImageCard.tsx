"use client";

import { useEffect, useRef, useState } from "react";
import TagChipEditor from "@/components/dataset/TagChipEditor";
import { datasetImageUrl } from "@/lib/api/datasets";

interface Props {
  datasetId: number;
  filename: string;
  initialTags: string[];
  onTagsSaved: (filename: string, tags: string[]) => void;
  onSave: (filename: string, tags: string[]) => Promise<void>;
}

export default function DatasetImageCard({
  datasetId,
  filename,
  initialTags,
  onTagsSaved,
  onSave,
}: Props) {
  const [tags, setTags] = useState(initialTags);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestTags = useRef(tags);

  useEffect(() => {
    setTags(initialTags);
  }, [initialTags]);

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

  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden flex flex-col">
      <div className="aspect-square bg-[var(--bg)] relative">
        <img
          src={datasetImageUrl(datasetId, filename)}
          alt={filename}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
      <div className="p-3 space-y-2 flex-1">
        <div className="text-xs text-[var(--muted)] truncate" title={filename}>
          {filename}
        </div>
        <TagChipEditor tags={tags} onChange={handleChange} disabled={saving} />
        <div className="text-[10px] text-[var(--muted)] min-h-[14px]">
          {saving ? "Saving…" : error ? <span className="text-red-400">{error}</span> : null}
        </div>
      </div>
    </div>
  );
}
