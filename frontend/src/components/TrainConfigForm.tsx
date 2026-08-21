"use client";

import type { GpuDefaultsInfo } from "@/lib/api/settings";
import { useTrainConfigForm } from "@/hooks/useTrainConfigForm";
import type { TrainConfig } from "@/lib/trainConfigSanitize";
import TrainCheckpointingSection from "@/components/train/TrainCheckpointingSection";
import TrainConceptsSection from "@/components/train/TrainConceptsSection";
import TrainHyperparametersSection from "@/components/train/TrainHyperparametersSection";
import TrainLoraSection from "@/components/train/TrainLoraSection";
import TrainModelSection from "@/components/train/TrainModelSection";
import TrainOptimizationSection from "@/components/train/TrainOptimizationSection";
import TrainOptimizerSection from "@/components/train/TrainOptimizerSection";
import TrainPerformanceSection from "@/components/train/TrainPerformanceSection";
import TrainTargetsSection from "@/components/train/TrainTargetsSection";

interface TrainConfigFormProps {
  config: TrainConfig;
  onChange: (config: TrainConfig) => void;
  gpuDefaults?: GpuDefaultsInfo;
}

export default function TrainConfigForm({ config, onChange, gpuDefaults }: TrainConfigFormProps) {
  const form = useTrainConfigForm(config, onChange);

  return (
    <div className="space-y-5">
      <TrainModelSection form={form} />
      <TrainLoraSection form={form} />
      <TrainTargetsSection form={form} />
      <TrainHyperparametersSection form={form} />
      <TrainOptimizerSection form={form} />
      <TrainConceptsSection form={form} />
      <TrainOptimizationSection form={form} gpuDefaults={gpuDefaults} />
      <TrainPerformanceSection form={form} />
      <TrainCheckpointingSection form={form} />
    </div>
  );
}
