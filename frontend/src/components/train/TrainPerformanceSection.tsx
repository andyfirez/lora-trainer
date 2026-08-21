"use client";

import FormSection from "@/components/ui/FormSection";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainCheckboxInput, TrainNumberInput } from "@/components/train/TrainFormFields";

interface TrainPerformanceSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainPerformanceSection({ form }: TrainPerformanceSectionProps) {
  const {
    config,
    set,
    cacheLatentsEnabled,
    textEncoderTrainingEnabled,
    cacheTextEncoderEnabled,
  } = form;

  return (
    <FormSection title="Performance">
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted">Caching</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
          <TrainCheckboxInput
            label="Cache Latents (RAM)"
            checked={cacheLatentsEnabled}
            onChange={(value) => set("cache_latents", value)}
            paramKey="cache_latents"
          />
          <TrainCheckboxInput
            label="Cache Text Encoder Outputs (RAM)"
            checked={cacheTextEncoderEnabled}
            onChange={(value) => set("cache_text_encoder_outputs", value)}
            disabled={textEncoderTrainingEnabled}
            paramKey="cache_text_encoder_outputs"
          />
          <div className="space-y-1">
            <TrainCheckboxInput
              label="Cache Latents to Disk (.npz)"
              checked={(config.cache_latents_to_disk as boolean | undefined) ?? false}
              onChange={(value) => set("cache_latents_to_disk", value)}
              disabled={!cacheLatentsEnabled}
              paramKey="cache_latents_to_disk"
            />
            {!cacheLatentsEnabled && <p className="text-xs text-muted">Requires RAM caching to be enabled.</p>}
          </div>
          <div className="space-y-1">
            <TrainCheckboxInput
              label="Cache Text Encoder Outputs to Disk"
              checked={(config.cache_text_encoder_outputs_to_disk as boolean | undefined) ?? false}
              onChange={(value) => set("cache_text_encoder_outputs_to_disk", value)}
              disabled={!cacheTextEncoderEnabled}
              paramKey="cache_text_encoder_outputs_to_disk"
            />
            {!cacheTextEncoderEnabled && !textEncoderTrainingEnabled && (
              <p className="text-xs text-muted">Requires RAM caching to be enabled.</p>
            )}
          </div>
        </div>
        {textEncoderTrainingEnabled && (
          <p className="text-xs text-muted mt-1">
            Text encoder output caching is disabled while training text encoders.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end mt-4">
        <div className="flex items-end pb-1">
          <TrainCheckboxInput
            label="torch.compile (slower start)"
            checked={(config.torch_compile as boolean | undefined) ?? false}
            onChange={(value) => set("torch_compile", value)}
            paramKey="torch_compile"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <TrainNumberInput
          label="DataLoader Workers (0 = main thread)"
          value={(config.num_dataloader_workers as number | null | undefined) ?? 0}
          onChange={(value) => set("num_dataloader_workers", value ?? 0)}
          min={0}
          placeholder="0"
          paramKey="num_dataloader_workers"
        />
        <div className="flex items-end pb-1">
          <TrainCheckboxInput
            label="Pin Memory (requires workers > 0)"
            checked={(config.dataloader_pin_memory as boolean | undefined) ?? true}
            onChange={(value) => set("dataloader_pin_memory", value)}
            paramKey="dataloader_pin_memory"
          />
        </div>
      </div>
    </FormSection>
  );
}
