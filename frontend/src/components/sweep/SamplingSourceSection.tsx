"use client";

import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import type { SweepParameter } from "@/lib/sweepUtils";

type Config = Record<string, unknown>;

interface SamplingSourceSectionProps {
  config: Config;
  onChange: (config: Config) => void;
}

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

export default function SamplingSourceSection({ config, onChange }: SamplingSourceSectionProps) {
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  const sourceType = (config.source_type as string) ?? "manual";
  const loraPaths = (config.lora_paths as string[]) ?? [];

  function updateLoraPath(i: number, value: string) {
    const next = loraPaths.map((p, idx) => (idx === i ? value : p));
    set("lora_paths", next);
  }

  function addLoraPath() {
    set("lora_paths", [...loraPaths, ""]);
  }

  function removeLoraPath(i: number) {
    set("lora_paths", loraPaths.filter((_, idx) => idx !== i));
  }

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
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted">LoRA paths (vary when multiple)</div>
          {loraPaths.map((path, i) => (
            <div key={i} className="flex items-center gap-2 min-w-0">
              <div className="flex-1 min-w-0">
                <PathInput
                  label=""
                  value={path}
                  onChange={(v) => updateLoraPath(i, v)}
                  placeholder="Path to .safetensors"
                  pickerTitle="Select LoRA"
                  kind="file"
                />
              </div>
              <button
                type="button"
                onClick={() => removeLoraPath(i)}
                className="p-1.5 rounded hover:bg-white/10 text-muted hover:text-error shrink-0"
              >
                <X size={13} />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addLoraPath}
            className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border rounded-lg px-3 py-2 w-full justify-center"
          >
            <Plus size={13} /> Add LoRA path
          </button>
        </div>
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

export function syncLoraPathsToParameters(config: Config): Config {
  const paths = ((config.lora_paths as string[]) ?? []).filter(Boolean);
  const parameters = { ...(config.parameters as Record<string, SweepParameter> | undefined) };
  if (paths.length) {
    parameters.lora_path = { mode: "vary", values: paths };
  }
  return { ...config, parameters };
}
