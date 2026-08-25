"use client";

import BaseModelSelect from "@/components/BaseModelSelect";
import PathInput from "@/components/PathInput";
import PlaygroundGenerateControls from "@/components/playground/PlaygroundGenerateControls";
import type { GeneratePrimaryLabel } from "@/hooks/useSamplingJobs";
import {
  EXPERIMENT_PARAM_LABELS,
  getParameters,
  setParameter,
} from "@/lib/experimentUtils";

interface PlaygroundSourceStripProps {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
  primaryLabel: GeneratePrimaryLabel;
  primaryBusy: boolean;
  onPrimary: () => void;
  error: string | null;
}

export default function PlaygroundSourceStrip({
  config,
  onChange,
  primaryLabel,
  primaryBusy,
  onPrimary,
  error,
}: PlaygroundSourceStripProps) {
  const parameters = getParameters(config);
  const baseModel = String(parameters.base_model_name?.value ?? "");

  return (
    <div className="border-b border-border px-3 py-2">
      <div className="grid items-start gap-3 lg:grid-cols-[minmax(14rem,18rem)_minmax(0,1fr)_minmax(10rem,12rem)]">
        <BaseModelSelect
          label={EXPERIMENT_PARAM_LABELS.base_model_name}
          value={baseModel}
          onChange={(value) =>
            onChange(setParameter(config, "base_model_name", { mode: "fixed", value }))
          }
        />
        <PathInput
          label="Output Folder"
          value={(config.output_dir as string) ?? ""}
          onChange={(value) => onChange({ ...config, output_dir: value })}
          placeholder="output (project folder)"
          pickerTitle="Select Output Folder"
          kind="directory"
        />
        <PlaygroundGenerateControls
          primaryLabel={primaryLabel}
          primaryBusy={primaryBusy}
          onPrimary={onPrimary}
          error={error}
        />
      </div>
    </div>
  );
}
