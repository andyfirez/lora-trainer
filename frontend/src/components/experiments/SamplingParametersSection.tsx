"use client";

import { useState } from "react";
import ParamGroup from "@/components/experiments/ParamGroup";
import ExperimentField from "@/components/experiments/ExperimentField";
import { InheritedCheckboxField, InheritedSelectField } from "@/components/ui/InheritedGpuField";
import FormSection, { formSectionClass, formSectionTitleClass } from "@/components/ui/FormSection";
import { useSamplingOptions } from "@/hooks/useSamplingOptions";
import { labelClassName } from "@/components/ui/Input";
import {
  EXPERIMENT_PARAM_LABELS,
  type ExperimentParamKey,
  defaultExperimentParameter,
  getParameters,
  setParameter,
} from "@/lib/experimentUtils";
import type { GpuDefaultsInfo } from "@/lib/api/settings";
import { MIXED_PRECISION_OPTIONS, VAE_DTYPE_OPTIONS } from "@/lib/gpuConfigUtils";
import { useBaseModelOptions } from "@/hooks/useBaseModelOptions";

type Config = Record<string, unknown>;

interface SamplingParametersSectionProps {
  config: Config;
  onChange: (config: Config) => void;
  gpuDefaults?: GpuDefaultsInfo;
  allowVary?: boolean;
  hidePrompts?: boolean;
  hideSeed?: boolean;
}

function param(config: Config, key: ExperimentParamKey) {
  const parameters = getParameters(config);
  return (
    parameters[key] ??
    defaultExperimentParameter(
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
  allowVary = true,
  hidePrompts = false,
  hideSeed = false,
}: SamplingParametersSectionProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const { samplers, schedulers } = useSamplingOptions();
  const baseModelParam = param(config, "base_model_name");
  const { options: baseModelOptions, isLoading: baseModelsLoading } = useBaseModelOptions();

  function set(key: string, value: unknown) {
    const next: Config = { ...config };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  }

  function updateParam(key: ExperimentParamKey, value: ReturnType<typeof param>) {
    onChange(setParameter(config, key, value));
  }

  return (
    <>
      <FormSection title="Parameters">
        <div className="space-y-6">
          <ExperimentField
            label={EXPERIMENT_PARAM_LABELS.base_model_name}
            param={baseModelParam}
            onChange={(p) => updateParam("base_model_name", p)}
            type="select"
            selectOptions={
              baseModelsLoading
                ? [{ value: String(baseModelParam.value ?? ""), label: "Loading models…" }]
                : baseModelOptions
            }
            allowVary={allowVary}
          />

          {hidePrompts ? null : (
            <ParamGroup title="Prompts">
              <ExperimentField
                label={EXPERIMENT_PARAM_LABELS.prompt}
                param={param(config, "prompt")}
                onChange={(p) => updateParam("prompt", p)}
                multiline
                placeholder="Prompt text"
                allowVary={allowVary}
              />
              <ExperimentField
                label={EXPERIMENT_PARAM_LABELS.negative_prompt}
                param={param(config, "negative_prompt")}
                onChange={(p) => updateParam("negative_prompt", p)}
                placeholder="low quality, blurry"
                allowVary={allowVary}
              />
            </ParamGroup>
          )}

          <ParamGroup title="LoRA">
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.lora_weight}
              param={param(config, "lora_weight")}
              onChange={(p) => updateParam("lora_weight", p)}
              type="number"
              allowVary={allowVary}
            />
          </ParamGroup>

          <ParamGroup title="Sampler">
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.steps}
              param={param(config, "steps")}
              onChange={(p) => updateParam("steps", p)}
              type="number"
              allowVary={allowVary}
            />
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.cfg_scale}
              param={param(config, "cfg_scale")}
              onChange={(p) => updateParam("cfg_scale", p)}
              type="number"
              allowVary={allowVary}
            />
            {hideSeed ? null : (
              <ExperimentField
                label={EXPERIMENT_PARAM_LABELS.seed}
                param={param(config, "seed")}
                onChange={(p) => updateParam("seed", p)}
                type="number"
                allowVary={allowVary}
              />
            )}
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.sampler}
              param={param(config, "sampler")}
              onChange={(p) => updateParam("sampler", p)}
              type="select"
              selectOptions={samplers}
              allowVary={allowVary}
            />
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.scheduler}
              param={param(config, "scheduler")}
              onChange={(p) => updateParam("scheduler", p)}
              type="select"
              selectOptions={schedulers}
              allowVary={allowVary}
            />
          </ParamGroup>

          <ParamGroup title="Resolution">
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.width}
              param={param(config, "width")}
              onChange={(p) => updateParam("width", p)}
              type="number"
              allowVary={allowVary}
            />
            <ExperimentField
              label={EXPERIMENT_PARAM_LABELS.height}
              param={param(config, "height")}
              onChange={(p) => updateParam("height", p)}
              type="number"
              allowVary={allowVary}
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
