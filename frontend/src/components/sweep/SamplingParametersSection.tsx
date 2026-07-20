"use client";

import { useState } from "react";
import PathInput from "@/components/PathInput";
import SweepField from "@/components/sweep/SweepField";
import { diffusersSchedulerOptions } from "@/lib/sampleSamplerOptions";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import {
  SWEEP_PARAM_LABELS,
  type SweepParamKey,
  defaultSweepParameter,
  getParameters,
  setParameter,
} from "@/lib/sweepUtils";

type Config = Record<string, unknown>;

interface SamplingParametersSectionProps {
  config: Config;
  onChange: (config: Config) => void;
}

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

function param(config: Config, key: SweepParamKey) {
  const parameters = getParameters(config);
  return parameters[key] ?? defaultSweepParameter(key === "steps" || key === "cfg_scale" || key === "lora_weight" || key === "seed" || key === "width" || key === "height" ? "number" : "string");
}

export default function SamplingParametersSection({ config, onChange }: SamplingParametersSectionProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  function updateParam(key: SweepParamKey, value: ReturnType<typeof param>) {
    onChange(setParameter(config, key, value));
  }

  return (
    <>
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Parameters</div>
        <div className="space-y-5">
          <PathInput
            label="Base Model (default fixed)"
            value={String(param(config, "base_model_name").value ?? config.base_model_name ?? "")}
            onChange={(v) => updateParam("base_model_name", { mode: "fixed", value: v })}
            placeholder="stabilityai/stable-diffusion-xl-base-1.0"
            pickerTitle="Select Base Model"
            kind="model"
          />
          <SweepField
            label={SWEEP_PARAM_LABELS.prompt}
            param={param(config, "prompt")}
            onChange={(p) => updateParam("prompt", p)}
            multiline
            placeholder="Prompt text"
          />
          <SweepField
            label={SWEEP_PARAM_LABELS.negative_prompt}
            param={param(config, "negative_prompt")}
            onChange={(p) => updateParam("negative_prompt", p)}
            placeholder="low quality, blurry"
          />
          <SweepField
            label={SWEEP_PARAM_LABELS.lora_weight}
            param={param(config, "lora_weight")}
            onChange={(p) => updateParam("lora_weight", p)}
            type="number"
          />
          <div className="grid grid-cols-2 gap-4">
            <SweepField
              label={SWEEP_PARAM_LABELS.steps}
              param={param(config, "steps")}
              onChange={(p) => updateParam("steps", p)}
              type="number"
            />
            <SweepField
              label={SWEEP_PARAM_LABELS.cfg_scale}
              param={param(config, "cfg_scale")}
              onChange={(p) => updateParam("cfg_scale", p)}
              type="number"
            />
            <SweepField
              label={SWEEP_PARAM_LABELS.width}
              param={param(config, "width")}
              onChange={(p) => updateParam("width", p)}
              type="number"
            />
            <SweepField
              label={SWEEP_PARAM_LABELS.height}
              param={param(config, "height")}
              onChange={(p) => updateParam("height", p)}
              type="number"
            />
            <SweepField
              label={SWEEP_PARAM_LABELS.seed}
              param={param(config, "seed")}
              onChange={(p) => updateParam("seed", p)}
              type="number"
            />
          </div>
          <SweepField
            label={SWEEP_PARAM_LABELS.scheduler}
            param={param(config, "scheduler")}
            onChange={(p) => updateParam("scheduler", p)}
            type="select"
            selectOptions={diffusersSchedulerOptions}
          />
        </div>
      </section>

      <section className={sectionClass}>
        <button
          type="button"
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className={`${sectionTitleClass} w-full text-left flex items-center justify-between`}
        >
          Advanced (performance)
          <span className="text-muted text-xs">{advancedOpen ? "▲" : "▼"}</span>
        </button>
        {advancedOpen && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClassName}>Attention</label>
              <select
                className={selectClassName}
                value={(config.attention_mechanism as string) ?? "sdpa"}
                onChange={(e) => set("attention_mechanism", e.target.value)}
              >
                <option value="xformers">xformers</option>
                <option value="sdpa">SDPA (PyTorch 2.x)</option>
                <option value="default">Default</option>
              </select>
            </div>
            <div>
              <label className={labelClassName}>Mixed Precision</label>
              <select
                className={selectClassName}
                value={(config.mixed_precision as string) ?? "float16"}
                onChange={(e) => set("mixed_precision", e.target.value)}
              >
                <option value="bfloat16">bfloat16</option>
                <option value="float16">float16</option>
                <option value="float32">float32</option>
              </select>
            </div>
            <div>
              <label className={labelClassName}>VAE Dtype</label>
              <select
                className={selectClassName}
                value={(config.vae_dtype as string) ?? "auto"}
                onChange={(e) => set("vae_dtype", e.target.value)}
              >
                <option value="auto">Auto</option>
                <option value="bfloat16">bfloat16</option>
                <option value="float16">float16</option>
                <option value="float32">float32</option>
              </select>
            </div>
            <label className="flex items-center gap-2 cursor-pointer col-span-2">
              <input
                type="checkbox"
                checked={(config.tf32 as boolean) ?? true}
                onChange={(e) => set("tf32", e.target.checked)}
              />
              <span className="text-sm">TF32 matmul</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer col-span-2">
              <input
                type="checkbox"
                checked={(config.sample_vae_tiling as boolean) ?? true}
                onChange={(e) => set("sample_vae_tiling", e.target.checked)}
              />
              <span className="text-sm">VAE tiling</span>
            </label>
          </div>
        )}
      </section>
    </>
  );
}
