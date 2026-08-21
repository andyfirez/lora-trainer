"use client";

import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { CopyMinus, Pencil, Sparkles } from "lucide-react";
import AutoTagModal from "@/components/dataset/AutoTagModal";
import EditDatasetModal from "@/components/dataset/EditDatasetModal";
import DatasetImageCard from "@/components/dataset/DatasetImageCard";
import ImageCropModal from "@/components/dataset/ImageCropModal";
import PreprocessPanel from "@/components/dataset/PreprocessPanel";
import TagFrequencyPanel from "@/components/dataset/TagFrequencyPanel";
import Button from "@/components/ui/Button";
import BackLink from "@/components/ui/BackLink";
import ErrorAlert from "@/components/ui/ErrorAlert";
import { useDatasetDetail } from "@/hooks/useDatasetDetail";
import type { ImagePreprocessState } from "@/types";

const PAGE_SIZE = 24;

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = Number(params.id);
  const [page, setPage] = useState(1);
  const [showAutoTag, setShowAutoTag] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [cropFilename, setCropFilename] = useState<string | null>(null);
  const [filterIncomplete, setFilterIncomplete] = useState(false);
  const [removingDuplicates, setRemovingDuplicates] = useState(false);

  const {
    dataset,
    datasetError,
    items,
    itemsLoading,
    tagStats,
    preprocessStatus,
    duplicatesInfo,
    taggingStatus,
    preparing,
    bakeError,
    handleSaveTags,
    handleTagsSaved,
    handleBulkAdd,
    handleBulkRemove,
    handleAutoTag,
    handleRemoveDuplicates,
    handleDeleteImage,
    handleDatasetSaved,
  } = useDatasetDetail(datasetId, { autoBakePaused: cropFilename != null });

  const filteredItems = useMemo(() => {
    if (!filterIncomplete) return items;
    return items.filter((item) => item.preprocess_state !== "ready");
  }, [items, filterIncomplete]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const pageItems = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return filteredItems.slice(start, start + PAGE_SIZE);
  }, [filteredItems, page]);

  const taggingActive = taggingStatus?.status === "running";

  const taggingBannerMessage = (() => {
    if (!taggingStatus || taggingStatus.status === "idle") return null;
    if (taggingStatus.status === "running") {
      return taggingStatus.total > 0
        ? `Auto-tagging in progress: ${taggingStatus.current}/${taggingStatus.total}.`
        : "Auto-tagging in progress…";
    }
    if (taggingStatus.status === "completed") return "Auto-tagging completed.";
    return taggingStatus.error ? `Auto-tagging failed: ${taggingStatus.error}` : "Auto-tagging failed.";
  })();

  const onDatasetSaved = async () => {
    setPage(1);
    await handleDatasetSaved();
  };

  const onRemoveDuplicates = async () => {
    const duplicateCount = duplicatesInfo?.duplicate_count ?? 0;
    if (duplicateCount <= 0) return;
    if (
      !confirm(
        `Remove ${duplicateCount} duplicate image${duplicateCount === 1 ? "" : "s"}? This cannot be undone.`,
      )
    ) {
      return;
    }
    setRemovingDuplicates(true);
    try {
      await handleRemoveDuplicates();
    } finally {
      setRemovingDuplicates(false);
    }
  };

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
          <BackLink href="/datasets" aria-label="Back to datasets" />
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-text font-display truncate">{dataset.name}</h1>
            <p className="text-xs text-muted truncate">{dataset.relative_path}</p>
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

      {bakeError && <ErrorAlert>{bakeError}</ErrorAlert>}

      {taggingBannerMessage && (
        <div
          className={`rounded-xl border px-4 py-3 flex flex-wrap items-center justify-between gap-3 ${
            taggingStatus?.status === "failed"
              ? "border-error/30 bg-error-muted"
              : taggingStatus?.status === "completed"
                ? "border-success/30 bg-success-muted"
                : "border-accent/30 bg-accent-muted"
          }`}
        >
          <span className="text-sm text-text">{taggingBannerMessage}</span>
        </div>
      )}

      <PreprocessPanel
        dataset={dataset}
        status={preprocessStatus}
        preparing={preparing}
        onUpdated={onDatasetSaved}
      />

      <TagFrequencyPanel
        tags={tagStats?.tags ?? []}
        onBulkAdd={handleBulkAdd}
        onBulkRemove={handleBulkRemove}
        disabled={taggingActive}
      />

      <div className="space-y-4">
        {(duplicatesInfo?.duplicate_count ?? 0) > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-warning/30 bg-warning-muted px-4 py-3">
            <span className="text-sm text-text">
              Found {duplicatesInfo?.duplicate_count} duplicate image
              {duplicatesInfo?.duplicate_count === 1 ? "" : "s"}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={removingDuplicates || taggingActive}
              onClick={() => void onRemoveDuplicates()}
            >
              <CopyMinus size={14} />
              {removingDuplicates ? "Removing…" : "Remove duplicates"}
            </Button>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
          <span>
            {filteredItems.length} images{filterIncomplete ? " (filtered)" : ""}
          </span>
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
              <Button variant="secondary" size="sm" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>
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
                onDelete={handleDeleteImage}
                deleteDisabled={taggingActive || preparing}
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
          onSaved={onDatasetSaved}
        />
      )}

      <EditDatasetModal open={showEdit} dataset={dataset} onClose={() => setShowEdit(false)} onSaved={onDatasetSaved} />

      <AutoTagModal open={showAutoTag} onClose={() => setShowAutoTag(false)} onSubmit={handleAutoTag} />
    </div>
  );
}
