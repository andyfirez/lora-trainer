"use client";

import useSWR from "swr";
import { useCallback, useEffect, useRef, useState } from "react";
import { datasetsApi } from "@/lib/api/datasets";
import type {
  AutotagStatusResponse,
  DatasetItem,
  DatasetItems,
  ImagePreprocessState,
  PreprocessStatus,
  TaggingMode,
} from "@/types";

export const CAPTION_EXTENSION = ".txt";

type PreprocessStateKey = ImagePreprocessState;

function decrementPreprocessStatus(
  status: PreprocessStatus,
  state: PreprocessStateKey | null | undefined,
): PreprocessStatus {
  const next: PreprocessStatus = {
    ...status,
    total: Math.max(0, status.total - 1),
  };
  if (state === "no_crop") next.no_crop = Math.max(0, next.no_crop - 1);
  else if (state === "stale") next.stale = Math.max(0, next.stale - 1);
  else if (state === "cropped") next.cropped = Math.max(0, next.cropped - 1);
  else if (state === "ready") next.ready = Math.max(0, next.ready - 1);
  next.preprocess_ready = next.total > 0 && next.ready === next.total;
  return next;
}

function removeItemsFromCache(
  current: DatasetItems | undefined,
  filenames: Set<string>,
): DatasetItems | undefined {
  if (!current) return current;
  return {
    ...current,
    items: current.items.filter((item) => !filenames.has(item.filename)),
  };
}

export function useDatasetDetail(datasetId: number, options?: { autoBakePaused?: boolean }) {
  const autoBakePaused = options?.autoBakePaused ?? false;
  const [preparing, setPreparing] = useState(false);
  const [bakeError, setBakeError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const bakingInFlight = useRef(false);
  const lastBakeKey = useRef<string | null>(null);

  const enabled = Number.isFinite(datasetId);

  const {
    data: dataset,
    error: datasetError,
    mutate: mutateDataset,
  } = useSWR(enabled ? `/datasets/${datasetId}` : null, () => datasetsApi.get(datasetId));

  const {
    data: itemsData,
    mutate: mutateItems,
    isLoading: itemsLoading,
  } = useSWR(enabled ? `/datasets/${datasetId}/items` : null, () =>
    datasetsApi.listItems(datasetId, CAPTION_EXTENSION),
  );

  const { data: tagStats, mutate: mutateStats } = useSWR(
    enabled ? `/datasets/${datasetId}/tags/stats` : null,
    () => datasetsApi.getTagStats(datasetId, CAPTION_EXTENSION),
  );

  const { data: preprocessStatus, mutate: mutatePreprocessStatus } = useSWR(
    enabled ? `/datasets/${datasetId}/preprocess/status` : null,
    () => datasetsApi.getPreprocessStatus(datasetId),
  );

  const { data: duplicatesInfo, mutate: mutateDuplicates } = useSWR(
    enabled ? `/datasets/${datasetId}/duplicates` : null,
    () => datasetsApi.getDuplicates(datasetId),
  );

  const { data: taggingStatus, mutate: mutateTaggingStatus } = useSWR<AutotagStatusResponse>(
    enabled ? `/datasets/${datasetId}/autotag/status` : null,
    () => datasetsApi.getAutotagStatus(datasetId),
    {
      refreshInterval: (latest) => (latest?.status === "running" ? 1500 : 0),
    },
  );

  const items = itemsData?.items ?? [];

  const refreshAll = useCallback(async () => {
    await Promise.all([
      mutateItems(),
      mutateStats(),
      mutatePreprocessStatus(),
      mutateDataset(),
      mutateDuplicates(),
    ]);
  }, [mutateItems, mutateStats, mutatePreprocessStatus, mutateDataset, mutateDuplicates]);

  const refreshWithoutDataset = useCallback(async () => {
    await Promise.all([mutateItems(), mutateStats(), mutatePreprocessStatus(), mutateDuplicates()]);
  }, [mutateItems, mutateStats, mutatePreprocessStatus, mutateDuplicates]);

  const resetBakeTracking = useCallback(() => {
    lastBakeKey.current = null;
  }, []);

  useEffect(() => {
    if (!enabled || autoBakePaused || !dataset?.target_resolution || !preprocessStatus) {
      return;
    }

    const incomplete = preprocessStatus.no_crop + preprocessStatus.cropped + preprocessStatus.stale;
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
    setBakeError(null);

    void (async () => {
      try {
        await datasetsApi.bakeAll(datasetId);
        lastBakeKey.current = null;
        await refreshAll();
      } catch (error) {
        setBakeError(error instanceof Error ? error.message : "Failed to bake images");
      } finally {
        bakingInFlight.current = false;
        setPreparing(false);
      }
    })();
  }, [autoBakePaused, dataset?.target_resolution, preprocessStatus, datasetId, enabled, refreshAll]);

  const prevTaggingStatus = useRef<string | null>(null);
  useEffect(() => {
    if (!taggingStatus) return;
    if (taggingStatus.status === "completed" && prevTaggingStatus.current === "running") {
      void mutateItems();
      void mutateStats();
    }
    prevTaggingStatus.current = taggingStatus.status;
  }, [taggingStatus, mutateItems, mutateStats]);

  const handleSaveTags = useCallback(
    async (filename: string, tags: string[]) => {
      await datasetsApi.updateCaption(datasetId, filename, tags, CAPTION_EXTENSION);
    },
    [datasetId],
  );

  const handleTagsSaved = useCallback(
    (filename: string, tags: string[]) => {
      void mutateItems(
        (current) =>
          current
            ? {
                ...current,
                items: current.items.map((item) =>
                  item.filename === filename ? { ...item, tags, has_caption: tags.length > 0 } : item,
                ),
              }
            : current,
        { revalidate: false },
      );
      void mutateStats();
    },
    [mutateItems, mutateStats],
  );

  const handleBulkAdd = useCallback(
    async (tag: string) => {
      await datasetsApi.bulkAddTag(datasetId, tag, undefined, CAPTION_EXTENSION);
      await refreshAll();
    },
    [datasetId, refreshAll],
  );

  const handleBulkRemove = useCallback(
    async (tag: string) => {
      await datasetsApi.bulkRemoveTag(datasetId, tag, undefined, CAPTION_EXTENSION);
      await refreshAll();
    },
    [datasetId, refreshAll],
  );

  const handleAutoTag = useCallback(
    async (opts: { mode: TaggingMode; threshold: number; model: string; strip_rating: boolean }) => {
      await datasetsApi.autotag(datasetId, {
        ...opts,
        caption_extension: CAPTION_EXTENSION,
      });
      void mutateTaggingStatus();
    },
    [datasetId, mutateTaggingStatus],
  );

  const handleRemoveDuplicates = useCallback(async () => {
    const previousItems = itemsData;
    const previousStatus = preprocessStatus;
    const previousDuplicates = duplicatesInfo;

    void mutateDuplicates({ duplicate_count: 0 }, { revalidate: false });

    try {
      await datasetsApi.removeDuplicates(datasetId, CAPTION_EXTENSION);
      await refreshWithoutDataset();
    } catch (error) {
      void mutateItems(previousItems, { revalidate: false });
      if (previousStatus) {
        void mutatePreprocessStatus(previousStatus, { revalidate: false });
      }
      void mutateDuplicates(previousDuplicates, { revalidate: false });
      setDeleteError(error instanceof Error ? error.message : "Failed to remove duplicates");
      throw error;
    }
  }, [
    datasetId,
    duplicatesInfo,
    itemsData,
    mutateDuplicates,
    mutateItems,
    mutatePreprocessStatus,
    preprocessStatus,
    refreshWithoutDataset,
  ]);

  const handleDeleteImage = useCallback(
    async (filename: string) => {
      const deletedItem = itemsData?.items.find((item) => item.filename === filename);
      const previousItems = itemsData;
      const previousStatus = preprocessStatus;

      setDeleteError(null);
      void mutateItems((current) => removeItemsFromCache(current, new Set([filename])), {
        revalidate: false,
      });
      if (preprocessStatus) {
        void mutatePreprocessStatus(
          decrementPreprocessStatus(
            preprocessStatus,
            (deletedItem?.preprocess_state as PreprocessStateKey | undefined) ?? "no_crop",
          ),
          { revalidate: false },
        );
      }

      try {
        await datasetsApi.deleteImage(datasetId, filename, CAPTION_EXTENSION);
        await Promise.all([mutateStats(), mutateDuplicates(), mutatePreprocessStatus()]);
      } catch (error) {
        void mutateItems(previousItems, { revalidate: false });
        if (previousStatus) {
          void mutatePreprocessStatus(previousStatus, { revalidate: false });
        }
        setDeleteError(error instanceof Error ? error.message : "Failed to delete image");
        throw error;
      }
    },
    [
      datasetId,
      itemsData,
      mutateDuplicates,
      mutateItems,
      mutatePreprocessStatus,
      mutateStats,
      preprocessStatus,
    ],
  );

  const handleDatasetSaved = useCallback(async () => {
    resetBakeTracking();
    await refreshAll();
  }, [refreshAll, resetBakeTracking]);

  const updateItemLocally = useCallback(
    (filename: string, updater: (item: DatasetItem) => DatasetItem) => {
      void mutateItems(
        (current) =>
          current
            ? {
                ...current,
                items: current.items.map((item) => (item.filename === filename ? updater(item) : item)),
              }
            : current,
        { revalidate: false },
      );
    },
    [mutateItems],
  );

  const handleCropSaved = useCallback(
    async (filename: string) => {
      resetBakeTracking();
      updateItemLocally(filename, (item) => ({ ...item, preprocess_state: "ready" }));
      await mutatePreprocessStatus();
    },
    [mutatePreprocessStatus, resetBakeTracking, updateItemLocally],
  );

  return {
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
    deleteError,
    refreshAll,
    resetBakeTracking,
    handleSaveTags,
    handleTagsSaved,
    handleBulkAdd,
    handleBulkRemove,
    handleAutoTag,
    handleRemoveDuplicates,
    handleDeleteImage,
    handleDatasetSaved,
    handleCropSaved,
    updateItemLocally,
  };
}
