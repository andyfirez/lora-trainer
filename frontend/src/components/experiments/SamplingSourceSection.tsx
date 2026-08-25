"use client";

import PathInput from "@/components/PathInput";
import LoraEntryField from "@/components/experiments/LoraEntryField";
import FormSection from "@/components/ui/FormSection";
import {
  EXPERIMENT_PARAM_LABELS,
  defaultExperimentParameter,
  getParameters,
  setParameter,
  syncLoraPathsToParameters,
} from "@/lib/experimentUtils";

type Config = Record<string, unknown>;

interface SamplingSourceSectionProps {
  config: Config;
  onChange: (config: Config) => void;
  allowVary?: boolean;
  showIncludeBase?: boolean;
}

function loraPathParam(config: Config) {
  const parameters = getParameters(config);
  return parameters.lora_path ?? defaultExperimentParameter("string");
}

export default function SamplingSourceSection({
  config,
  onChange,
  allowVary = true,
  showIncludeBase = true,
}: SamplingSourceSectionProps) {
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  function updateLoraPathParam(param: ReturnType<typeof loraPathParam>) {
    onChange(syncLoraPathsToParameters(setParameter(config, "lora_path", param)));
  }

  return (
    <FormSection title="Source">
      <LoraEntryField
        label={EXPERIMENT_PARAM_LABELS.lora_path}
        param={loraPathParam(config)}
        onChange={updateLoraPathParam}
        allowVary={allowVary}
      />
      {showIncludeBase ? (
        <label className="flex items-center gap-2 cursor-pointer text-sm">
          <input
            type="checkbox"
            checked={(config.include_base_model_sample as boolean) ?? false}
            onChange={(e) => set("include_base_model_sample", e.target.checked)}
          />
          Include base model (no LoRA) in experiment
        </label>
      ) : null}
      <PathInput
        label="Output Folder"
        value={(config.output_dir as string) ?? ""}
        onChange={(v) => set("output_dir", v)}
        placeholder="output (project folder)"
        pickerTitle="Select Output Folder"
        kind="directory"
      />
    </FormSection>
  );
}

export { syncLoraPathsToParameters };
