"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import ConfigForm from "@/components/ConfigForm";
import PageHeader from "@/components/ui/PageHeader";

function NewConfigPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const typeParam = searchParams.get("type");

  useEffect(() => {
    if (typeParam === "sampling") {
      router.replace("/sampling/new");
    }
  }, [typeParam, router]);

  if (typeParam === "sampling") {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Redirecting…
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader
        title="New Training Config"
        description="Configure and save a reusable SDXL LoRA training config"
      />
      <ConfigForm configType="training" />
    </div>
  );
}

export default function NewConfigPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewConfigPageContent />
    </Suspense>
  );
}
