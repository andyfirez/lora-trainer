"use client";

import { useState } from "react";
import ParamGroup from "@/components/sweep/ParamGroup";
import SweepField from "@/components/sweep/SweepField";
import SweepPathField from "@/components/sweep/SweepPathField";
import { InheritedCheckboxField, InheritedSelectField } from "@/components/ui/InheritedGpuField";
import FormSection, { formSectionClass, formSectionTitleClass } from "@/components/ui/FormSection";
import { diffusersSchedulerOptions } from "@/lib/sampleSamplerOptions";
import { labelClassName } from "@/components/ui/Input";
import {
  SWEEP_PARAM_LABELS,
  type SweepParamKey,
  defaultSweepParameter,
  getParameters,
  setParameter,
} from "@/lib/sweepUtils";
import type { GpuDefaultsInfo } from "@/lib/api/settings";
import { MIXED_PRECISION_OPTIONS, VAE_DTYPE_OPTIONS } from "@/lib/gpuConfigUtils";

type Config = Record<string, unknown>;

interface SamplingParametersSectionProps {
  config: Config;
  onChange: (config: Config) => void;
  gpuDefaults?: GpuDefaultsInfo;
}

function param(config: Config, key: SweepParamKey) {
  const parameters = getParameters(config);
  return (
    parameters[key] ??
    defaultSweepParameter(
      key === "steps" ||
        key === "cfg_scale" ||
        key === "lora_weight" ||
        key === "seed" ||
        key === "width" ||
        key === "height"
        ? "number"
        : "string",
    )
  );
}

export default function SamplingParametersSection({
  config,
  onChange,
  gpuDefaults,
}: SamplingParametersSectionProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  function set(key: string, value: unknown) {
    const next: Config = { ...config };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  }

  function updateParam(key: SweepParamKey, value: ReturnType<typeof param>) {
    onChange(setParameter(config, key, value));
  }

  return (
    <>
      <FormSection title="Parameters">
        <div className="space-y-6">
          <SweepPathField
            label={SWEEP_PARAM_LABELS.base_model_name}
            param={param(config, "base_model_name")}
            onChange={(p) => updateParam("base_model_name", p)}
            placeholder="stabilityai/stable-diffusion-xl-base-1.0"
            pickerTitle="Select Base Model"
            kind="model"
          />

          <ParamGroup title="Prompts">
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
          </ParamGroup>

          <ParamGroup title="LoRA">
            <SweepField
              label={SWEEP_PARAM_LABELS.lora_weight}
              param={param(config, "lora_weight")}
              onChange={(p) => updateParam("lora_weight", p)}
              type="number"
            />
          </ParamGroup>

          <ParamGroup title="Sampler">
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
              label={SWEEP_PARAM_LABELS.seed}
              param={param(config, "seed")}
              onChange={(p) => updateParam("seed", p)}
              type="number"
            />
            <SweepField
              label={SWEEP_PARAM_LABELS.scheduler}
              param={param(config, "scheduler")}
              onChange={(p) => updateParam("scheduler", p)}
              type="select"
              selectOptions={diffusersSchedulerOptions}
            />
          </ParamGroup>

          <ParamGroup title="Resolution">
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
          </ParamGroup>
        </div>
      </FormSection>

      <section className={formSectionClass}>
        <button
          type="button"
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className={`${formSectionTitleClass} w-full text-left flex items-center justify-between`}
        >
          Advanced (performance)
          <span className="text-muted text-xs">{advancedOpen ? "▲" : "▼"}</span>
        </button>
        {advancedOpen && (
          <div className="space-y-4">
            {gpuDefaults ? (
              <>
                <InheritedSelectField
                  label="Mixed Precision"
                  value={config.mixed_precision as string | undefined}
                  globalDefault={gpuDefaults.mixed_precision}
                  options={MIXED_PRECISION_OPTIONS}
                  onChange={(v) => set("mixed_precision", v)}
                  paramKey="mixed_precision"
                />
                <InheritedSelectField
                  label="VAE Dtype"
                  value={config.vae_dtype as string | undefined}
                  globalDefault={gpuDefaults.vae_dtype}
                  options={VAE_DTYPE_OPTIONS}
                  onChange={(v) => set("vae_dtype", v)}
                  paramKey="vae_dtype"
                />
                <InheritedCheckboxField
                  label="VAE tiling"
                  value={config.sample_vae_tiling as boolean | undefined}
                  globalDefault={gpuDefaults.sample_vae_tiling}
                  onChange={(v) => set("sample_vae_tiling", v)}
                  paramKey="sample_vae_tiling"
                />
              </>
            ) : (
              <p className="text-sm text-muted">Loading GPU defaults…</p>
            )}
          </div>
        )}
      </section>
    </>
  );
}
