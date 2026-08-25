"use client";

import SamplingParametersSection from "@/components/experiments/SamplingParametersSection";
import SamplingSourceSection, { syncLoraPathsToParameters } from "@/components/experiments/SamplingSourceSection";
import GridLayoutSection from "@/components/experiments/GridLayoutSection";
import ExperimentPreviewBar from "@/components/experiments/ExperimentPreviewBar";
import type { GpuDefaultsInfo } from "@/lib/api/settings";

type Config = Record<string, unknown>;

interface SamplingConfigFormProps {
  config: Config;
  onChange: (config: Config) => void;
  gpuDefaults?: GpuDefaultsInfo;
  readOnly?: boolean;
}

export default function SamplingConfigForm({
  config,
  onChange,
  gpuDefaults,
  readOnly = false,
}: SamplingConfigFormProps) {
  function handleChange(next: Config) {
    if (readOnly) return;
    const synced = syncLoraPathsToParameters(next);
    onChange(synced);
  }

  return (
    <div className={readOnly ? "pointer-events-none opacity-80 space-y-5 pb-16" : "space-y-5 pb-16"}>
      <SamplingSourceSection config={config} onChange={handleChange} />
      <SamplingParametersSection config={config} onChange={handleChange} gpuDefaults={gpuDefaults} />
      <GridLayoutSection config={config} onChange={handleChange} />
      <ExperimentPreviewBar config={config} />
    </div>
  );
}
