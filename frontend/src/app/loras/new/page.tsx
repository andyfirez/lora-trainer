"use client";

import { Suspense } from "react";
import { Loader2 } from "lucide-react";
import RunnableCreatePage from "@/components/runnable/RunnableCreatePage";
import { lorasApi } from "@/lib/api/loras";
import { TrainConfig } from "@/lib/defaultConfig";
import TrainConfigForm from "@/components/TrainConfigForm";

function validateTrainConfig(config: Record<string, unknown>): string | null {
  const concepts = config.concepts;
  if (!Array.isArray(concepts)) return null;
  for (let i = 0; i < concepts.length; i++) {
    const concept = concepts[i];
    if (!concept || typeof concept !== "object" || (concept as Record<string, unknown>).dataset_id == null) {
      return `Concept ${i + 1}: select a dataset`;
    }
  }
  return null;
}

function NewLoraPageContent() {
  return (
    <RunnableCreatePage
      title="New LoRA"
      description="Configure and start a new SDXL LoRA training run"
      nameLabel="LoRA Name"
      namePlaceholder="my-sdxl-lora"
      entityLabel="LoRA"
      defaultYaml={TrainConfig.DEFAULT_YAML}
      redirectTo={(id) => `/loras/${id}`}
      validate={validateTrainConfig}
      create={({ name, configYaml, enqueue }) =>
        lorasApi.create({ name, config_yaml: configYaml }).then(async (lora) => (enqueue ? lorasApi.enqueue(lora.id) : lora))
      }
      renderForm={(config, onChange, gpuDefaults) => (
        <TrainConfigForm config={config} onChange={onChange} gpuDefaults={gpuDefaults} />
      )}
    />
  );
}

export default function NewLoraPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewLoraPageContent />
    </Suspense>
  );
}
