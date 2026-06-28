"use client";

import useSWR from "swr";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, Sparkles } from "lucide-react";
import AutoTagModal from "@/components/dataset/AutoTagModal";
import DatasetImageCard from "@/components/dataset/DatasetImageCard";
import TagFrequencyPanel from "@/components/dataset/TagFrequencyPanel";
import { datasetsApi } from "@/lib/api/datasets";
import { jobsApi } from "@/lib/api/jobs";
import type { DatasetItem, Job, TaggingMode } from "@/types";

const PAGE_SIZE = 24;
const CAPTION_EXTENSION = ".txt";

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = Number(params.id);
  const [page, setPage] = useState(1);
  const [showAutoTag, setShowAutoTag] = useState(false);
  const [taggingJobId, setTaggingJobId] = useState<number | null>(null);
  const [localItems, setLocalItems] = useState<DatasetItem[]>([]);

  const { data: dataset, error: datasetError } = useSWR(
    Number.isFinite(datasetId) ? `/datasets/${datasetId}` : null,
    () => datasetsApi.get(datasetId)
  );

  const {
    data: itemsData,
    mutate: mutateItems,
    isLoading: itemsLoading,
  } = useSWR(Number.isFinite(datasetId) ? `/datasets/${datasetId}/items` : null, () =>
    datasetsApi.listItems(datasetId, CAPTION_EXTENSION)
  );

  const { data: tagStats, mutate: mutateStats } = useSWR(
    Number.isFinite(datasetId) ? `/datasets/${datasetId}/tags/stats` : null,
    () => datasetsApi.getTagStats(datasetId, CAPTION_EXTENSION)
  );

  const { data: taggingJob } = useSWR<Job | null>(
    taggingJobId ? `/jobs/${taggingJobId}` : null,
    () => (taggingJobId ? jobsApi.get(taggingJobId) : null),
    {
      refreshInterval: (latest) => {
        if (!latest) return 0;
        if (["completed", "failed", "cancelled"].includes(latest.status)) return 0;
        return 2000;
      },
    }
  );

  useEffect(() => {
    if (itemsData?.items) {
      setLocalItems(itemsData.items);
    }
  }, [itemsData]);

  useEffect(() => {
    if (!taggingJob) return;
    if (["completed", "failed", "cancelled"].includes(taggingJob.status)) {
      void mutateItems();
      void mutateStats();
    }
  }, [taggingJob, mutateItems, mutateStats]);

  const totalPages = Math.max(1, Math.ceil(localItems.length / PAGE_SIZE));
  const pageItems = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return localItems.slice(start, start + PAGE_SIZE);
  }, [localItems, page]);

  const refreshAll = useCallback(async () => {
    await Promise.all([mutateItems(), mutateStats()]);
  }, [mutateItems, mutateStats]);

  const handleSaveTags = useCallback(
    async (filename: string, tags: string[]) => {
      await datasetsApi.updateCaption(datasetId, filename, tags, CAPTION_EXTENSION);
    },
    [datasetId]
  );

  const handleTagsSaved = useCallback(
    (filename: string, tags: string[]) => {
      setLocalItems((prev) =>
        prev.map((item) =>
          item.filename === filename ? { ...item, tags, has_caption: tags.length > 0 } : item
        )
      );
      void mutateStats();
    },
    [mutateStats]
  );

  const handleBulkAdd = useCallback(
    async (tag: string) => {
      await datasetsApi.bulkAddTag(datasetId, tag, undefined, CAPTION_EXTENSION);
      await refreshAll();
    },
    [datasetId, refreshAll]
  );

  const handleBulkRemove = useCallback(
    async (tag: string) => {
      await datasetsApi.bulkRemoveTag(datasetId, tag, undefined, CAPTION_EXTENSION);
      await refreshAll();
    },
    [datasetId, refreshAll]
  );

  const handleAutoTag = useCallback(
    async (options: { mode: TaggingMode; threshold: number; model: string; strip_rating: boolean }) => {
      const result = await datasetsApi.autotag(datasetId, {
        ...options,
        caption_extension: CAPTION_EXTENSION,
        enqueue: true,
      });
      setTaggingJobId(result.job_id);
    },
    [datasetId]
  );

  const taggingActive =
    taggingJob != null && ["pending", "queued", "running"].includes(taggingJob.status);

  const taggingBannerMessage = (() => {
    if (!taggingJobId) return null;
    if (!taggingJob) return `Auto-tagging job #${taggingJobId} started.`;
    if (taggingJob.status === "completed") return `Auto-tagging job #${taggingJobId} completed.`;
    if (taggingJob.status === "failed") {
      return taggingJob.error_message
        ? `Auto-tagging job #${taggingJobId} failed: ${taggingJob.error_message}`
        : `Auto-tagging job #${taggingJobId} failed.`;
    }
    if (taggingJob.status === "cancelled") return `Auto-tagging job #${taggingJobId} was cancelled.`;
    return `Auto-tagging job #${taggingJobId} is ${taggingJob.status}.`;
  })();

  if (datasetError) {
    return <div className="text-red-400">Failed to load dataset</div>;
  }

  if (!dataset) {
    return <div className="text-[var(--muted)]">Loading dataset…</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <Link
            href="/datasets"
            className="p-2 rounded-lg border border-[var(--border)] text-[var(--muted)] hover:text-white hover:bg-white/5"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-white truncate">{dataset.name}</h1>
            <p className="text-xs text-[var(--muted)] truncate">{dataset.image_dir}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowAutoTag(true)}
          disabled={taggingActive}
          className="flex items-center gap-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          <Sparkles size={15} />
          Auto-tag
        </button>
      </div>

      {taggingJobId && taggingBannerMessage && (
        <div
          className={`rounded-xl border px-4 py-3 flex flex-wrap items-center justify-between gap-3 ${
            taggingJob?.status === "failed"
              ? "border-red-400/30 bg-red-400/10"
              : taggingJob?.status === "completed"
                ? "border-green-400/30 bg-green-400/10"
                : "border-[var(--accent)]/30 bg-[var(--accent)]/10"
          }`}
        >
          <span className="text-sm text-white">{taggingBannerMessage}</span>
          <Link
            href={`/jobs/${taggingJobId}`}
            className="shrink-0 text-sm font-medium text-[var(--accent)] hover:text-[var(--accent-hover)]"
          >
            Open job →
          </Link>
        </div>
      )}

      <TagFrequencyPanel
        tags={tagStats?.tags ?? []}
        onBulkAdd={handleBulkAdd}
        onBulkRemove={handleBulkRemove}
        disabled={taggingActive}
      />

      <div className="space-y-4">
        <div className="flex items-center justify-between text-sm text-[var(--muted)]">
          <span>{localItems.length} images</span>
          {totalPages > 1 && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((value) => value - 1)}
                className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-40"
              >
                Prev
              </button>
              <span>
                {page} / {totalPages}
              </span>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((value) => value + 1)}
                className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </div>

        {itemsLoading ? (
          <div className="text-[var(--muted)]">Loading images…</div>
        ) : localItems.length === 0 ? (
          <div className="text-center py-16 text-[var(--muted)] border border-dashed border-[var(--border)] rounded-xl">
            No images found in this directory.
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {pageItems.map((item) => (
              <DatasetImageCard
                key={item.filename}
                datasetId={datasetId}
                filename={item.filename}
                initialTags={item.tags}
                onSave={handleSaveTags}
                onTagsSaved={handleTagsSaved}
              />
            ))}
          </div>
        )}
      </div>

      <AutoTagModal
        open={showAutoTag}
        onClose={() => setShowAutoTag(false)}
        onSubmit={handleAutoTag}
      />
    </div>
  );
}
