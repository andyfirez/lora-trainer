"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import SamplingConfigForm from "@/components/SamplingConfigForm";
import BackLink from "@/components/ui/BackLink";
import PageHeader from "@/components/ui/PageHeader";
import { useExperiments } from "@/hooks/useExperiments";
import { samplingsApi } from "@/lib/api/samplings";
import { settingsApi } from "@/lib/api/settings";
import {
  defaultExperimentConfig,
  loadExperimentDraft,
  saveExperimentDraft,
} from "@/lib/experimentState";
import { syncLoraPathsToParameters } from "@/lib/experimentUtils";
import { validatePlaygroundConfig } from "@/lib/playgroundState";

export default function ExperimentsNewPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const copyFromId = Number(searchParams.get("copyFrom") ?? "");
  const [hydrated, setHydrated] = useState(false);
  const [config, setConfig] = useState<Record<string, unknown>>(defaultExperimentConfig());

  const { submitting, error, setError, createAndEnqueue } = useExperiments();
  const { data: settingsData } = useSWR("/settings", () => settingsApi.get());
  const { data: copySource } = useSWR(
    Number.isFinite(copyFromId) && copyFromId > 0 ? `/samplings/${copyFromId}` : null,
    () => samplingsApi.get(copyFromId),
  );

  useEffect(() => {
    if (Number.isFinite(copyFromId) && copyFromId > 0) {
      if (!copySource) return;
      if (copySource.kind === "experiment") {
        setConfig(structuredClone(copySource.config));
      }
      setHydrated(true);
      return;
    }
    setConfig(loadExperimentDraft().config);
    setHydrated(true);
  }, [copyFromId, copySource]);

  useEffect(() => {
    if (!hydrated) return;
    saveExperimentDraft({ config });
  }, [hydrated, config]);

  const handleConfigChange = useCallback(
    (next: Record<string, unknown>) => {
      setConfig(syncLoraPathsToParameters(next));
      setError(null);
    },
    [setError],
  );

  const handleCreate = async () => {
    const validationError = validatePlaygroundConfig(config);
    if (validationError) {
      setError(validationError);
      return;
    }
    try {
      const created = await createAndEnqueue(config);
      if (created?.id) {
        router.push(`/experiments/${created.id}`);
      } else {
        router.push("/experiments");
      }
    } catch {
      // error handled in hook
    }
  };

  if (!hydrated) {
    return <div className="p-8 text-sm text-muted">Loading…</div>;
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-center gap-3">
        <BackLink href="/experiments" aria-label="Back to experiments" />
        <PageHeader
          title={copySource ? "Copy experiment" : "New experiment"}
          description="Configure parameters and enqueue a grid run"
        />
      </div>

      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={() => router.push("/experiments")}
          className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-muted hover:text-text"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={() => void handleCreate()}
          disabled={submitting}
          className="rounded-lg bg-sampling px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {submitting ? "Creating…" : "Create & Enqueue"}
        </button>
      </div>

      {error && <p className="text-sm text-error">{error}</p>}

      <SamplingConfigForm
        config={config}
        onChange={handleConfigChange}
        gpuDefaults={settingsData?.gpu_defaults}
      />
    </div>
  );
}
