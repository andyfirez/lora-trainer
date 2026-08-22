"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import type { QueueEntry } from "@/components/playground/queueTypes";
import { lorasApi } from "@/lib/api/loras";
import { samplingsApi } from "@/lib/api/samplings";
import { getParameters, loraPathsFromParameter, syncLoraPathsToParameters } from "@/lib/sweepUtils";
import type { SamplingResponse } from "@/types";

const ACTIVE = new Set(["queued", "running"]);

export type GeneratePrimaryLabel = "Generate" | "Enqueue" | "Interrupt";

function samplingPromptLabel(sampling: SamplingResponse): string {
  try {
    const prompt = getParameters(sampling.config).prompt;
    const value = prompt?.mode === "vary" ? prompt.values?.[0] : prompt?.value;
    const text = String(value ?? "").trim();
    return text.slice(0, 80) || sampling.name || "Sampling";
  } catch {
    return sampling.name || "Sampling";
  }
}

export function useSamplingJobs() {
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [sessionIds, setSessionIds] = useState<number[]>([]);

  const { data: samplings, mutate: mutateSamplings } = useSWR(
    "/samplings",
    () => samplingsApi.list(),
    { refreshInterval: 1000 },
  );
  const { data: loras, mutate: mutateLoras } = useSWR("/loras", () => lorasApi.list(), {
    refreshInterval: 1000,
  });

  const queueItems: QueueEntry[] = useMemo(() => {
    const samplingItems = (samplings ?? [])
      .filter((item) => ACTIVE.has(item.status))
      .map((item) => ({
        kind: "sampling" as const,
        id: item.id,
        label: samplingPromptLabel(item),
        status: item.status,
      }));
    const loraItems = (loras ?? [])
      .filter((item) => ACTIVE.has(item.status))
      .map((item) => ({
        kind: "lora" as const,
        id: item.id,
        label: item.name,
        status: item.status,
      }));
    return [...loraItems, ...samplingItems];
  }, [samplings, loras]);

  const anyRunning = queueItems.some((item) => item.status === "running");
  const myRunning = (samplings ?? []).find(
    (item) => item.status === "running" && sessionIds.includes(item.id),
  );
  const primaryLabel: GeneratePrimaryLabel = myRunning
    ? "Interrupt"
    : anyRunning
      ? "Enqueue"
      : "Generate";

  const interrupt = async () => {
    if (!myRunning) return;
    await samplingsApi.cancel(myRunning.id);
    await mutateSamplings();
  };

  const generate = async (config: Record<string, unknown>, options: { composeGrids: boolean }) => {
    setSubmitting(true);
    setError(null);
    try {
      const synced = syncLoraPathsToParameters(config);
      const withFlags = { ...synced, compose_grids: options.composeGrids };
      const loraPaths = loraPathsFromParameter(getParameters(synced).lora_path);
      const created = await samplingsApi.generate(withFlags, loraPaths);
      setSessionIds((ids) => [...ids, created.id]);
      await mutateSamplings();
      return created;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start sampling");
      return null;
    } finally {
      setSubmitting(false);
    }
  };

  const cancelQueueItem = async (item: QueueEntry) => {
    if (item.kind === "sampling") {
      await samplingsApi.cancel(item.id);
      await mutateSamplings();
      return;
    }
    await lorasApi.cancel(item.id, false);
    await mutateLoras();
  };

  return {
    samplings,
    sessionIds,
    queueItems,
    myRunning,
    primaryLabel,
    submitting,
    error,
    setError,
    interrupt,
    generate,
    cancelQueueItem,
  };
}
