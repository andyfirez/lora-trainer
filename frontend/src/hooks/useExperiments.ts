"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { samplingsApi } from "@/lib/api/samplings";
import { getParameters, loraPathsFromParameter, syncLoraPathsToParameters } from "@/lib/experimentUtils";
import type { SamplingResponse } from "@/types";

const ACTIVE = new Set(["queued", "running"]);
const TERMINAL = new Set(["completed", "failed", "cancelled"]);

function experimentLabel(sampling: SamplingResponse): string {
  try {
    const prompt = getParameters(sampling.config).prompt;
    const value = prompt?.mode === "vary" ? prompt.values?.[0] : prompt?.value;
    const text = String(value ?? "").trim();
    return text.slice(0, 80) || sampling.name || "Experiment";
  } catch {
    return sampling.name || "Experiment";
  }
}

export function useExperiments() {
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const { data: experiments, mutate: mutateExperiments } = useSWR(
    "/samplings?kind=experiment",
    () => samplingsApi.listExperiments(),
    { refreshInterval: 1000 },
  );

  const inProgress = useMemo(
    () => (experiments ?? []).filter((item) => ACTIVE.has(item.status)),
    [experiments],
  );

  const completed = useMemo(
    () => (experiments ?? []).filter((item) => TERMINAL.has(item.status)),
    [experiments],
  );

  const createAndEnqueue = async (config: Record<string, unknown>) => {
    setSubmitting(true);
    setError(null);
    try {
      const synced = syncLoraPathsToParameters({ ...config, compose_grids: true });
      const loraPaths = loraPathsFromParameter(getParameters(synced).lora_path);
      const label = experimentLabel({ config: synced, name: "Experiment" } as SamplingResponse);
      const created = await samplingsApi.generate(synced, loraPaths, {
        kind: "experiment",
        name: label.slice(0, 80) || "Experiment",
        composeGrids: true,
      });
      await mutateExperiments();
      return created;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create experiment");
      throw err;
    } finally {
      setSubmitting(false);
    }
  };

  const repeatExperiment = async (id: number) => {
    setError(null);
    try {
      await samplingsApi.enqueue(id);
      await mutateExperiments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to repeat experiment");
      throw err;
    }
  };

  return {
    experiments,
    inProgress,
    completed,
    submitting,
    error,
    setError,
    createAndEnqueue,
    repeatExperiment,
    mutateExperiments,
  };
}
