"use client";

import { useCallback, useEffect, useState } from "react";
import useSWR from "swr";
import PlaygroundGenerateControls from "@/components/playground/PlaygroundGenerateControls";
import PlaygroundQueue from "@/components/playground/PlaygroundQueue";
import SamplingConfigForm from "@/components/SamplingConfigForm";
import { useSamplingJobs } from "@/hooks/useSamplingJobs";
import { settingsApi } from "@/lib/api/settings";
import { validatePlaygroundConfig } from "@/lib/playgroundState";
import { loadSweepState, saveSweepState } from "@/lib/sweepState";
import { syncLoraPathsToParameters } from "@/lib/sweepUtils";

export default function SweepPage() {
  const [hydrated, setHydrated] = useState(false);
  const [config, setConfig] = useState<Record<string, unknown>>({});

  const {
    queueItems,
    primaryLabel,
    submitting,
    error,
    setError,
    interrupt,
    generate,
    cancelQueueItem,
  } = useSamplingJobs();

  const { data: settingsData } = useSWR("/settings", () => settingsApi.get());

  useEffect(() => {
    setConfig(loadSweepState().config);
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    saveSweepState({ config });
  }, [hydrated, config]);

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
    await generate(config, { composeGrids: true });
  };

  if (!hydrated) {
    return <div className="p-8 text-sm text-muted">Loading sweep…</div>;
  }

  return (
    <div className="-m-4 flex h-[100dvh] flex-col overflow-hidden bg-bg md:-m-6 lg:-m-8">
      <div className="border-b border-border px-3 py-2">
        <PlaygroundGenerateControls
          className="max-w-xs"
          primaryLabel={primaryLabel}
          primaryBusy={submitting}
          onPrimary={() => void handlePrimary()}
          error={error}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pt-4">
        <div className="mx-auto max-w-4xl">
          <SamplingConfigForm
            config={config}
            onChange={handleConfigChange}
            gpuDefaults={settingsData?.gpu_defaults}
          />
        </div>
      </div>
      <PlaygroundQueue items={queueItems} onCancel={(item) => void cancelQueueItem(item)} />
    </div>
  );
}
