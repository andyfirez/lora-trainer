"use client";

import FormSection from "@/components/ui/FormSection";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainNumberInput, TrainSelectInput, lrSchedulerOptions } from "@/components/train/TrainFormFields";

interface TrainHyperparametersSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainHyperparametersSection({ form }: TrainHyperparametersSectionProps) {
  const { config, set } = form;

  return (
    <FormSection title="Training">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <TrainNumberInput
          label="Epochs"
          value={config.epochs as number | null | undefined}
          onChange={(value) => set("epochs", value)}
          min={1}
          placeholder="30"
          paramKey="epochs"
        />
        <TrainNumberInput
          label="Batch Size"
          value={config.batch_size as number | null | undefined}
          onChange={(value) => set("batch_size", value)}
          min={1}
          placeholder="1"
          paramKey="batch_size"
        />
        <TrainNumberInput
          label="Grad Accumulation Steps"
          value={config.gradient_accumulation_steps as number | null | undefined}
          onChange={(value) => set("gradient_accumulation_steps", value)}
          min={1}
          placeholder="1"
          paramKey="gradient_accumulation_steps"
        />
        <TrainSelectInput
          label="LR Scheduler"
          value={(config.lr_scheduler as string) ?? "constant"}
          onChange={(value) => set("lr_scheduler", value)}
          options={lrSchedulerOptions}
          paramKey="lr_scheduler"
        />
        <TrainNumberInput
          label="LR Warmup Steps"
          value={config.lr_warmup_steps as number | null | undefined}
          onChange={(value) => set("lr_warmup_steps", value)}
          min={0}
          placeholder="0"
          paramKey="lr_warmup_steps"
        />
        <TrainNumberInput
          label="Min SNR Gamma"
          value={config.min_snr_gamma as number | null | undefined}
          onChange={(value) => set("min_snr_gamma", value)}
          min={0}
          step={0.5}
          placeholder="5"
          paramKey="min_snr_gamma"
        />
        <TrainNumberInput
          label="Noise Offset"
          value={config.noise_offset as number | null | undefined}
          onChange={(value) => set("noise_offset", value)}
          min={0}
          step={0.001}
          placeholder="0.0357"
          paramKey="noise_offset"
        />
      </div>
    </FormSection>
  );
}
