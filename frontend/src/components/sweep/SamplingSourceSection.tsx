"use client";

import PathInput from "@/components/PathInput";
import SweepPathField from "@/components/sweep/SweepPathField";
import { inputClassName, labelClassName } from "@/components/ui/Input";
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

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

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

  const sourceType = (config.source_type as string) ?? "manual";

  return (
    <section className={sectionClass}>
      <div className={sectionTitleClass}>Source</div>
      <div className="flex gap-4">
        <label className="flex items-center gap-2 cursor-pointer text-sm">
          <input
            type="radio"
            checked={sourceType === "manual"}
            onChange={() => set("source_type", "manual")}
          />
          Manual LoRA paths
        </label>
        <label className="flex items-center gap-2 cursor-pointer text-sm">
          <input
            type="radio"
            checked={sourceType === "training_job"}
            onChange={() => set("source_type", "training_job")}
          />
          From training job
        </label>
      </div>
      {sourceType === "manual" ? (
        <SweepPathField
          label={SWEEP_PARAM_LABELS.lora_path}
          param={loraPathParam(config)}
          onChange={updateLoraPathParam}
        />
      ) : (
        <div className="space-y-3">
          <div>
            <label className={labelClassName}>Training job ID</label>
            <input
              type="number"
              className={inputClassName}
              value={(config.source_job_id as number | null) ?? ""}
              placeholder="Set when launching from job"
              onChange={(e) =>
                set("source_job_id", e.target.value === "" ? null : Number(e.target.value))
              }
            />
            <p className="text-xs text-muted mt-1">
              Checkpoints are resolved automatically when the job runs.
            </p>
          </div>
          <label className="flex items-center gap-2 cursor-pointer text-sm">
            <input
              type="checkbox"
              checked={(config.include_final_checkpoint as boolean) ?? true}
              onChange={(e) => set("include_final_checkpoint", e.target.checked)}
            />
            Include final checkpoint
          </label>
        </div>
      )}
      <PathInput
        label="Output Folder"
        value={(config.output_dir as string) ?? ""}
        onChange={(v) => set("output_dir", v)}
        placeholder="D:\loras\output"
        pickerTitle="Select Output Folder"
        kind="directory"
      />
    </section>
  );
}

export { syncLoraPathsToParameters };
