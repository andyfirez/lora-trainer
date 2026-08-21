"use client";

import PathInput from "@/components/PathInput";
import LoraEntryField from "@/components/sweep/LoraEntryField";
import FormSection from "@/components/ui/FormSection";
import {
  SWEEP_PARAM_LABELS,
  defaultSweepParameter,
  getParameters,
  setParameter,
  syncLoraPathsToParameters,
} from "@/lib/sweepUtils";

type Config = Record<string, unknown>;

interface SamplingSourceSectionProps {
  config: Config;
  onChange: (config: Config) => void;
}

function loraPathParam(config: Config) {
  const parameters = getParameters(config);
  return parameters.lora_path ?? defaultSweepParameter("string");
}

export default function SamplingSourceSection({ config, onChange }: SamplingSourceSectionProps) {
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  function updateLoraPathParam(param: ReturnType<typeof loraPathParam>) {
    onChange(syncLoraPathsToParameters(setParameter(config, "lora_path", param)));
  }

  return (
    <FormSection title="Source">
      <LoraEntryField
        label={SWEEP_PARAM_LABELS.lora_path}
        param={loraPathParam(config)}
        onChange={updateLoraPathParam}
      />
      <label className="flex items-center gap-2 cursor-pointer text-sm">
        <input
          type="checkbox"
          checked={(config.include_base_model_sample as boolean) ?? false}
          onChange={(e) => set("include_base_model_sample", e.target.checked)}
        />
        Include base model (no LoRA) in sweep
      </label>
      <PathInput
        label="Output Folder"
        value={(config.output_dir as string) ?? ""}
        onChange={(v) => set("output_dir", v)}
        placeholder="D:\loras\output"
        pickerTitle="Select Output Folder"
        kind="directory"
      />
    </FormSection>
  );
}

export { syncLoraPathsToParameters };
