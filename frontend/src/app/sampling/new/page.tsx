"use client";

import { Suspense } from "react";
import { Loader2 } from "lucide-react";
import RunnableCreatePage from "@/components/runnable/RunnableCreatePage";
import { samplingsApi } from "@/lib/api/samplings";
import { SamplingConfig } from "@/lib/defaultConfig";
import SamplingConfigForm from "@/components/SamplingConfigForm";

function NewSamplingPageContent() {
  return (
    <RunnableCreatePage
      title="New Sampling"
      description="Configure and start a new sampling run"
      nameLabel="Sampling Name"
      namePlaceholder="my-sampling-run"
      entityLabel="Sampling"
      defaultYaml={SamplingConfig.DEFAULT_YAML}
      redirectTo={(id) => `/sampling/${id}`}
      create={({ name, configYaml, enqueue }) =>
        samplingsApi
          .create({ name, config_yaml: configYaml })
          .then(async (sampling) => (enqueue ? samplingsApi.enqueue(sampling.id) : sampling))
      }
      renderForm={(config, onChange, gpuDefaults) => (
        <SamplingConfigForm config={config} onChange={onChange} gpuDefaults={gpuDefaults} />
      )}
    />
  );
}

export default function NewSamplingPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewSamplingPageContent />
    </Suspense>
  );
}
