"use client";

import FormSection from "@/components/ui/FormSection";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainNumberInput } from "@/components/train/TrainFormFields";

interface TrainLoraSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainLoraSection({ form }: TrainLoraSectionProps) {
  const { config, set } = form;

  return (
    <FormSection title="LoRA">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <TrainNumberInput
          label="Rank"
          value={config.lora_rank as number | null | undefined}
          onChange={(value) => set("lora_rank", value)}
          min={1}
          max={256}
          placeholder="32"
          paramKey="lora_rank"
        />
        <TrainNumberInput
          label="Alpha"
          value={config.lora_alpha as number | null | undefined}
          onChange={(value) => set("lora_alpha", value)}
          min={0}
          step={0.1}
          placeholder="32.0"
          paramKey="lora_alpha"
        />
        <TrainNumberInput
          label="Dropout"
          value={config.lora_dropout as number | null | undefined}
          onChange={(value) => set("lora_dropout", value)}
          min={0}
          max={0.999}
          step={0.01}
          placeholder="0.0"
          paramKey="lora_dropout"
        />
      </div>
    </FormSection>
  );
}
