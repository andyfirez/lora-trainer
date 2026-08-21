"use client";

import FormSection from "@/components/ui/FormSection";
import { optimizerOptions } from "@/lib/optimizerPresets";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainCheckboxInput, TrainNumberInput, TrainSelectInput } from "@/components/train/TrainFormFields";

interface TrainOptimizerSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainOptimizerSection({ form }: TrainOptimizerSectionProps) {
  const {
    config,
    setNested,
    setOptimizerType,
    optimizerType,
    isAdamFamily,
    isAdafactor,
    isProdigy,
  } = form;

  return (
    <FormSection title="Optimizer">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <TrainSelectInput
          label="Type"
          value={optimizerType}
          onChange={(value) => setOptimizerType(value as typeof optimizerType)}
          options={[...optimizerOptions]}
          paramKey="optimizer.type"
        />
        {(isAdamFamily || isProdigy) && (
          <>
            <TrainNumberInput
              label="Weight Decay"
              value={(config.optimizer as Record<string, unknown> | undefined)?.weight_decay as
                | number
                | null
                | undefined}
              onChange={(value) => setNested("optimizer", "weight_decay", value)}
              min={0}
              step={0.01}
              placeholder="0.01"
              paramKey="optimizer.weight_decay"
            />
            <TrainNumberInput
              label="Beta 1"
              value={(config.optimizer as Record<string, unknown> | undefined)?.beta1 as number | null | undefined}
              onChange={(value) => setNested("optimizer", "beta1", value)}
              min={0}
              max={0.999}
              step={0.01}
              placeholder="0.9"
              paramKey="optimizer.beta1"
            />
            <TrainNumberInput
              label="Beta 2"
              value={(config.optimizer as Record<string, unknown> | undefined)?.beta2 as number | null | undefined}
              onChange={(value) => setNested("optimizer", "beta2", value)}
              min={0}
              max={0.999}
              step={0.001}
              placeholder="0.999"
              paramKey="optimizer.beta2"
            />
          </>
        )}
        {isAdafactor && (
          <>
            <TrainCheckboxInput
              label="Relative Step"
              checked={Boolean((config.optimizer as Record<string, unknown> | undefined)?.relative_step)}
              onChange={(value) => setNested("optimizer", "relative_step", value)}
              paramKey="optimizer.relative_step"
            />
            <TrainCheckboxInput
              label="Scale Parameter"
              checked={Boolean((config.optimizer as Record<string, unknown> | undefined)?.scale_parameter)}
              onChange={(value) => setNested("optimizer", "scale_parameter", value)}
              paramKey="optimizer.scale_parameter"
            />
            <TrainCheckboxInput
              label="Warmup Init"
              checked={Boolean((config.optimizer as Record<string, unknown> | undefined)?.warmup_init)}
              onChange={(value) => setNested("optimizer", "warmup_init", value)}
              paramKey="optimizer.warmup_init"
            />
          </>
        )}
        {isProdigy && (
          <>
            <TrainCheckboxInput
              label="Decouple"
              checked={((config.optimizer as Record<string, unknown> | undefined)?.decouple as boolean) ?? true}
              onChange={(value) => setNested("optimizer", "decouple", value)}
              paramKey="optimizer.decouple"
            />
            <TrainCheckboxInput
              label="Use Bias Correction"
              checked={
                ((config.optimizer as Record<string, unknown> | undefined)?.use_bias_correction as boolean) ?? true
              }
              onChange={(value) => setNested("optimizer", "use_bias_correction", value)}
              paramKey="optimizer.use_bias_correction"
            />
            <TrainCheckboxInput
              label="Safeguard Warmup"
              checked={
                ((config.optimizer as Record<string, unknown> | undefined)?.safeguard_warmup as boolean) ?? true
              }
              onChange={(value) => setNested("optimizer", "safeguard_warmup", value)}
              paramKey="optimizer.safeguard_warmup"
            />
            <TrainNumberInput
              label="d0"
              value={(config.optimizer as Record<string, unknown> | undefined)?.d0 as number | null | undefined}
              onChange={(value) => setNested("optimizer", "d0", value)}
              min={0}
              step={0.00001}
              placeholder="0.00001"
              paramKey="optimizer.d0"
            />
            <TrainNumberInput
              label="d Coef"
              value={(config.optimizer as Record<string, unknown> | undefined)?.d_coef as number | null | undefined}
              onChange={(value) => setNested("optimizer", "d_coef", value)}
              min={0}
              step={0.1}
              placeholder="1.0"
              paramKey="optimizer.d_coef"
            />
          </>
        )}
      </div>
    </FormSection>
  );
}
