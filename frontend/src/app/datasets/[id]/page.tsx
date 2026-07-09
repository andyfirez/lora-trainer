"use client";

import useSWR from "swr";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Pencil, Sparkles } from "lucide-react";
import AutoTagModal from "@/components/dataset/AutoTagModal";
import EditDatasetModal from "@/components/dataset/EditDatasetModal";
import DatasetImageCard from "@/components/dataset/DatasetImageCard";
import ImageCropModal from "@/components/dataset/ImageCropModal";
import PreprocessPanel from "@/components/dataset/PreprocessPanel";
import TagFrequencyPanel from "@/components/dataset/TagFrequencyPanel";
import Button from "@/components/ui/Button";
import { datasetsApi } from "@/lib/api/datasets";
import { jobsApi } from "@/lib/api/jobs";
import type { DatasetItem, ImagePreprocessState, Job, TaggingMode } from "@/types";

const PAGE_SIZE = 24;
const CAPTION_EXTENSION = ".txt";

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = Number(params.id);
  const [page, setPage] = useState(1);
  const [showAutoTag, setShowAutoTag] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [taggingJobId, setTaggingJobId] = useState<number | null>(null);
  const [localItems, setLocalItems] = useState<DatasetItem[]>([]);
  const [cropFilename, setCropFilename] = useState<string | null>(null);
  const [filterIncomplete, setFilterIncomplete] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const bakingInFlight = useRef(false);
  const lastBakeKey = useRef<string | null>(null);

  const {
    data: dataset,
    error: datasetError,
    mutate: mutateDataset,
  } = useSWR(Number.isFinite(datasetId) ? `/datasets/${datasetId}` : null, () =>
    datasetsApi.get(datasetId)
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

  const { data: preprocessStatus, mutate: mutatePreprocessStatus } = useSWR(
    Number.isFinite(datasetId) ? `/datasets/${datasetId}/preprocess/status` : null,
    () => datasetsApi.getPreprocessStatus(datasetId)
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

  const refreshAll = useCallback(async () => {
    await Promise.all([mutateItems(), mutateStats(), mutatePreprocessStatus(), mutateDataset()]);
  }, [mutateItems, mutateStats, mutatePreprocessStatus, mutateDataset]);

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

  const handleDatasetSaved = useCallback(async () => {
    setPage(1);
    lastBakeKey.current = null;
    await refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!Number.isFinite(datasetId) || !dataset?.target_resolution || !preprocessStatus || cropFilename) {
      return;
    }

    const incomplete =
      preprocessStatus.no_crop + preprocessStatus.cropped + preprocessStatus.stale;
    if (incomplete <= 0) {
      lastBakeKey.current = null;
      return;
    }
    if (bakingInFlight.current) return;

    const bakeKey = [
      dataset.target_resolution,
      preprocessStatus.total,
      preprocessStatus.no_crop,
      preprocessStatus.cropped,
      preprocessStatus.stale,
    ].join(":");
    if (lastBakeKey.current === bakeKey) return;

    lastBakeKey.current = bakeKey;
    bakingInFlight.current = true;
    setPreparing(true);

    void (async () => {
      try {
        await datasetsApi.bakeAll(datasetId);
        lastBakeKey.current = null;
        await refreshAll();
      } catch {
        // Status refresh will reflect partial progress; same bakeKey prevents retry loops.
      } finally {
        bakingInFlight.current = false;
        setPreparing(false);
      }
    })();
  }, [dataset?.target_resolution, preprocessStatus, cropFilename, datasetId, refreshAll]);

  const filteredItems = useMemo(() => {
    if (!filterIncomplete) return localItems;
    return localItems.filter((item) => item.preprocess_state !== "ready");
  }, [localItems, filterIncomplete]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const pageItems = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return filteredItems.slice(start, start + PAGE_SIZE);
  }, [filteredItems, page]);

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
    return <div className="text-error">Failed to load dataset</div>;
  }

  if (!dataset) {
    return <div className="text-muted">Loading dataset…</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <Link
            href="/datasets"
            className="p-2 rounded-lg border border-border text-muted hover:text-text hover:bg-white/5"
            aria-label="Back to datasets"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-text font-display truncate">{dataset.name}</h1>
            <p className="text-xs text-muted truncate">{dataset.image_dir}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => setShowEdit(true)}>
            <Pencil size={15} />
            Edit
          </Button>
          <Button onClick={() => setShowAutoTag(true)} disabled={taggingActive}>
            <Sparkles size={15} />
            Auto-tag
          </Button>
        </div>
      </div>

      {taggingJobId && taggingBannerMessage && (
        <div
          className={`rounded-xl border px-4 py-3 flex flex-wrap items-center justify-between gap-3 ${
            taggingJob?.status === "failed"
              ? "border-error/30 bg-error-muted"
              : taggingJob?.status === "completed"
                ? "border-success/30 bg-success-muted"
                : "border-accent/30 bg-accent-muted"
          }`}
        >
          <span className="text-sm text-text">{taggingBannerMessage}</span>
          <Link
            href={`/jobs/${taggingJobId}`}
            className="shrink-0 text-sm font-medium text-accent hover:text-accent-hover"
          >
            Open job →
          </Link>
        </div>
      )}

      <PreprocessPanel
        dataset={dataset}
        status={preprocessStatus}
        preparing={preparing}
        onUpdated={handleDatasetSaved}
      />

      <TagFrequencyPanel
        tags={tagStats?.tags ?? []}
        onBulkAdd={handleBulkAdd}
        onBulkRemove={handleBulkRemove}
        disabled={taggingActive}
      />

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
          <span>{filteredItems.length} images{filterIncomplete ? " (filtered)" : ""}</span>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input
              type="checkbox"
              checked={filterIncomplete}
              onChange={(e) => {
                setFilterIncomplete(e.target.checked);
                setPage(1);
              }}
              className="accent-accent"
            />
            Show only incomplete
          </label>
          {totalPages > 1 && (
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((value) => value - 1)}
              >
                Prev
              </Button>
              <span>
                {page} / {totalPages}
              </span>
              <Button
                variant="secondary"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((value) => value + 1)}
              >
                Next
              </Button>
            </div>
          )}
        </div>

        {itemsLoading ? (
          <div className="text-muted">Loading images…</div>
        ) : filteredItems.length === 0 ? (
          <div className="text-center py-16 text-muted border border-dashed border-border rounded-xl">
            {filterIncomplete ? "All images are ready." : "No images found in this directory."}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {pageItems.map((item) => (
              <DatasetImageCard
                key={item.filename}
                datasetId={datasetId}
                filename={item.filename}
                initialTags={item.tags}
                preprocessState={item.preprocess_state as ImagePreprocessState | undefined}
                canCrop={dataset.target_resolution != null}
                preparing={preparing}
                cacheKey={dataset.updated_at}
                onCropClick={() => setCropFilename(item.filename)}
                onSave={handleSaveTags}
                onTagsSaved={handleTagsSaved}
              />
            ))}
          </div>
        )}
      </div>

      {cropFilename && dataset.target_resolution != null && (
        <ImageCropModal
          datasetId={datasetId}
          filename={cropFilename}
          targetResolution={dataset.target_resolution}
          onClose={() => setCropFilename(null)}
          onSaved={handleDatasetSaved}
        />
      )}

      <EditDatasetModal
        open={showEdit}
        dataset={dataset}
        onClose={() => setShowEdit(false)}
        onSaved={handleDatasetSaved}
      />

      <AutoTagModal
        open={showAutoTag}
        onClose={() => setShowAutoTag(false)}
        onSubmit={handleAutoTag}
      />
    </div>
  );
}
