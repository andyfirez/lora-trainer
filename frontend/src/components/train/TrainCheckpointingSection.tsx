"use client";

import FormSection from "@/components/ui/FormSection";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainCheckboxInput, TrainNumberInput } from "@/components/train/TrainFormFields";

interface TrainCheckpointingSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainCheckpointingSection({ form }: TrainCheckpointingSectionProps) {
  const { config, set, checkpointingEnabled } = form;

  return (
    <FormSection title="Checkpointing">
      <div className="space-y-4">
        <TrainCheckboxInput
          label="Enable intermediate checkpoints"
          checked={checkpointingEnabled}
          onChange={(value) => set("checkpointing_enabled", value)}
          paramKey="checkpointing_enabled"
        />
        <TrainNumberInput
          label="Save Every N Epochs"
          value={config.save_every_n_epochs as number | null | undefined}
          onChange={(value) => set("save_every_n_epochs", value)}
          min={1}
          placeholder="1"
          disabled={!checkpointingEnabled}
          paramKey="save_every_n_epochs"
        />
      </div>
    </FormSection>
  );
}
