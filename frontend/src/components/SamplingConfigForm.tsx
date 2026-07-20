"use client";

import SamplingParametersSection from "@/components/sweep/SamplingParametersSection";
import SamplingSourceSection, { syncLoraPathsToParameters } from "@/components/sweep/SamplingSourceSection";
import GridLayoutSection from "@/components/sweep/GridLayoutSection";
import SweepPreviewBar from "@/components/sweep/SweepPreviewBar";

type Config = Record<string, unknown>;

interface SamplingConfigFormProps {
  config: Config;
  onChange: (config: Config) => void;
}

export default function SamplingConfigForm({ config, onChange }: SamplingConfigFormProps) {
  function handleChange(next: Config) {
    const synced = syncLoraPathsToParameters(next);
    onChange(synced);
  }

  return (
    <div className="space-y-5 pb-16">
      <SamplingSourceSection config={config} onChange={handleChange} />
      <SamplingParametersSection config={config} onChange={handleChange} />
      <GridLayoutSection config={config} onChange={handleChange} />
      <SweepPreviewBar config={config} />
    </div>
  );
}
