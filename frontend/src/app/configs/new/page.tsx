"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import ConfigForm from "@/components/ConfigForm";
import type { ConfigType } from "@/types";

function NewConfigPageContent() {
  const searchParams = useSearchParams();
  const typeParam = searchParams.get("type");
  const configType: ConfigType = typeParam === "sampling" ? "sampling" : "training";

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-text">
          New {configType === "training" ? "Training" : "Sampling"} Config
        </h1>
        <p className="text-text-muted mt-1">
          {configType === "training"
            ? "Configure and save a reusable SDXL LoRA training config"
            : "Configure and save a reusable sampling config"}
        </p>
      </div>
      <ConfigForm configType={configType} />
    </div>
  );
}

export default function NewConfigPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewConfigPageContent />
    </Suspense>
  );
}
