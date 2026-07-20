"use client";

import ConfigForm from "@/components/ConfigForm";
import PageHeader from "@/components/ui/PageHeader";

export default function NewSamplingPage() {
  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader
        title="New Sampling Config"
        description="Configure parameter sweep — vary prompts, LoRA weights, checkpoints, and more"
      />
      <ConfigForm configType="sampling" saveRedirectBase="/sampling" />
    </div>
  );
}
