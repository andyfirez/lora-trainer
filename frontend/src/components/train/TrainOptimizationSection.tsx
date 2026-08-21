"use client";

import FormSection from "@/components/ui/FormSection";
import { InheritedSelectField } from "@/components/ui/InheritedGpuField";
import type { GpuDefaultsInfo } from "@/lib/api/settings";
import { MIXED_PRECISION_OPTIONS, VAE_DTYPE_OPTIONS } from "@/lib/gpuConfigUtils";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import {
  TrainCheckboxInput,
  TrainNumberInput,
  TrainSelectInput,
  weightDtypeOptions,
} from "@/components/train/TrainFormFields";

interface TrainOptimizationSectionProps {
  form: TrainConfigFormContext;
  gpuDefaults?: GpuDefaultsInfo;
}

export default function TrainOptimizationSection({ form, gpuDefaults }: TrainOptimizationSectionProps) {
  const { config, set } = form;

  return (
    <FormSection title="Optimization">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        {gpuDefaults ? (
          <>
            <InheritedSelectField
              label="Mixed Precision"
              value={config.mixed_precision as string | undefined}
              globalDefault={gpuDefaults.mixed_precision}
              options={MIXED_PRECISION_OPTIONS}
              onChange={(value) => set("mixed_precision", value)}
              paramKey="mixed_precision"
            />
            <InheritedSelectField
              label="VAE Dtype"
              value={config.vae_dtype as string | undefined}
              globalDefault={gpuDefaults.vae_dtype}
              options={VAE_DTYPE_OPTIONS}
              onChange={(value) => set("vae_dtype", value)}
              paramKey="vae_dtype"
            />
          </>
        ) : (
          <TrainSelectInput
            label="Mixed Precision"
            value={(config.mixed_precision as string) ?? "float16"}
            onChange={(value) => set("mixed_precision", value)}
            options={weightDtypeOptions}
            paramKey="mixed_precision"
          />
        )}
        <TrainNumberInput
          label="Seed (optional)"
          value={(config.seed as number | null | undefined) ?? null}
          onChange={(value) => set("seed", value)}
          min={0}
          placeholder="random"
          paramKey="seed"
        />
        <div className="flex items-center pb-1">
          <TrainCheckboxInput
            label="Gradient Checkpointing"
            checked={(config.gradient_checkpointing as boolean | undefined) ?? true}
            onChange={(value) => set("gradient_checkpointing", value)}
            paramKey="gradient_checkpointing"
          />
        </div>
      </div>
    </FormSection>
  );
}
