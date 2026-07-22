"use client";

import { Suspense } from "react";
import { Loader2 } from "lucide-react";
import ConfigForm from "@/components/ConfigForm";
import PageHeader from "@/components/ui/PageHeader";

function NewTrainingPageContent() {
  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader
        title="New training config"
        description="Configure and save a reusable SDXL LoRA training config"
      />
      <ConfigForm configType="training" saveRedirectBase="/trainings" />
    </div>
  );
}

export default function NewTrainingPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewTrainingPageContent />
    </Suspense>
  );
}
