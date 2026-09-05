"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import PlaygroundPromptBar from "@/components/playground/PlaygroundPromptBar";
import PlaygroundCompactSettings from "@/components/playground/PlaygroundCompactSettings";
import PlaygroundGallery from "@/components/playground/PlaygroundGallery";
import PlaygroundLogsPanel from "@/components/playground/PlaygroundLogsPanel";
import PlaygroundProgressBar from "@/components/playground/PlaygroundProgressBar";
import PlaygroundSourceStrip from "@/components/playground/PlaygroundSourceStrip";
import ResizeSplit from "@/components/playground/ResizeSplit";
import ErrorAlert from "@/components/ui/ErrorAlert";
import type { GalleryEntry } from "@/components/playground/galleryTypes";
import { useSamplingJobs } from "@/hooks/useSamplingJobs";
import { samplingsApi } from "@/lib/api/samplings";
import { settingsApi } from "@/lib/api/settings";
import { mediaUrl } from "@/lib/media";
import {
  applyBatchCount,
  formatParamsInfotext,
  loadPlaygroundState,
  randomSeed,
  savePlaygroundState,
  seedFromParams,
  validatePlaygroundConfig,
} from "@/lib/playgroundState";
import { syncLoraPathsToParameters } from "@/lib/experimentUtils";
import type { RunnableSample } from "@/types";

function sampleEntry(samplingId: number, sample: RunnableSample): GalleryEntry {
  const params = (sample.metadata?.params as Record<string, unknown> | undefined) ?? undefined;
  return {
    key: `${samplingId}:${sample.path}`,
    url: mediaUrl(sample.url),
    kind: sample.kind === "grid" ? "grid" : "cell",
    samplingId,
    infotext: params ? formatParamsInfotext(params) : "",
    seed: seedFromParams(params),
    params,
  };
}

export default function SamplingPlayground() {
  const [hydrated, setHydrated] = useState(false);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [batchCount, setBatchCount] = useState(1);
  const [currentImage, setCurrentImage] = useState<GalleryEntry | null>(null);
  const [lastSeed, setLastSeed] = useState<number | null>(null);

  const {
    samplings,
    sessionIds,
    sessionReady,
    myRunning,
    showProgress,
    primaryLabel,
    primaryBusy,
    error,
    setError,
    beginImmediate,
    interrupt,
    generate,
  } = useSamplingJobs();

  const { data: settingsData } = useSWR("/settings", () => settingsApi.get());

  useEffect(() => {
    const stored = loadPlaygroundState();
    setConfig(stored.config);
    setBatchCount(stored.batchCount);
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || !sessionReady) return;
    savePlaygroundState({ config, batchCount, sessionIds });
  }, [hydrated, sessionReady, config, batchCount, sessionIds]);

  const progressPct =
    myRunning?.progress_step != null && myRunning.progress_total
      ? Math.round((myRunning.progress_step / myRunning.progress_total) * 100)
      : null;

  const logJob = useMemo(() => {
    const session = (samplings ?? []).filter((item) => sessionIds.includes(item.id));
    return (
      myRunning ??
      session.find((item) => item.status === "queued" || item.status === "running") ??
      session.at(-1)
    );
  }, [samplings, sessionIds, myRunning]);

  const jobError = logJob?.status === "failed" ? logJob.error_message : null;

  useEffect(() => {
    if (!myRunning || (myRunning.progress_step ?? 0) <= 0) return;
    setCurrentImage({
      key: `preview:${myRunning.id}`,
      url: samplingsApi.livePreviewUrl(myRunning.id, myRunning.progress_step ?? 0),
      kind: "preview",
      samplingId: myRunning.id,
      infotext: myRunning.progress_status ?? "Starting…",
      seed: null,
    });
  }, [myRunning]);

  useEffect(() => {
    if (!sessionIds.length) return;
    let cancelled = false;
    const load = async () => {
      const relevant = (samplings ?? []).filter(
        (item) => sessionIds.includes(item.id) && item.status !== "draft" && item.status !== "queued",
      );
      const batches = await Promise.all(
        relevant.map(async (item) => {
          try {
            const response = await samplingsApi.getSamples(item.id);
            return response.samples.map((sample) => sampleEntry(item.id, sample));
          } catch {
            return [];
          }
        }),
      );
      if (cancelled) return;
      const next = batches.flat();
      const latest = next.at(-1) ?? null;
      const latestSeed = latest?.seed ?? null;
      if (latestSeed != null) setLastSeed(latestSeed);
      if (!myRunning && latest) {
        setCurrentImage(latest);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [sessionIds, samplings, myRunning]);

  const handleConfigChange = useCallback(
    (next: Record<string, unknown>) => {
      setConfig(syncLoraPathsToParameters(next));
      setError(null);
    },
    [setError],
  );

  const handlePrimary = async () => {
    if (primaryLabel === "Interrupt") {
      await interrupt();
      return;
    }
    const validationError = validatePlaygroundConfig(config);
    if (validationError) {
      setError(validationError);
      return;
    }
    beginImmediate();
    setCurrentImage(null);
    const payload = batchCount > 1 ? applyBatchCount(config, batchCount, randomSeed) : config;
    await generate(payload, { composeGrids: false });
  };

  if (!hydrated) {
    return <div className="p-8 text-sm text-muted">Loading playground…</div>;
  }

  const generateControls = {
    primaryLabel,
    primaryBusy,
    onPrimary: () => void handlePrimary(),
    error,
  };

  return (
    <div className="-m-4 flex h-[100dvh] flex-col overflow-hidden bg-bg md:-m-6 lg:-m-8">
      <PlaygroundSourceStrip config={config} onChange={handleConfigChange} {...generateControls} />
      <div className="border-b border-border bg-surface">
        <PlaygroundProgressBar
          label={showProgress ? (myRunning?.progress_status ?? "Starting…") : null}
          pct={showProgress ? progressPct : null}
        />
        {jobError ? (
          <div className="px-3 pb-2">
            <ErrorAlert compact>
              <strong>Sampling failed:</strong> {jobError}
            </ErrorAlert>
          </div>
        ) : null}
        <PlaygroundLogsPanel job={logJob} />
      </div>
      <ResizeSplit
        left={
          <div className="flex min-h-0 flex-col">
            <PlaygroundPromptBar config={config} onChange={handleConfigChange} />
            <PlaygroundCompactSettings
              config={config}
              onChange={handleConfigChange}
              batchCount={batchCount}
              onBatchCountChange={setBatchCount}
              gpuDefaults={settingsData?.gpu_defaults}
              lastSeed={lastSeed}
            />
          </div>
        }
        right={<PlaygroundGallery item={currentImage} />}
      />
    </div>
  );
}
