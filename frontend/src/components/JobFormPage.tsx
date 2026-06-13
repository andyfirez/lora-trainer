"use client";

import { Suspense } from "react";
import useSWR from "swr";
import { useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import JobForm from "@/components/JobForm";
import { jobsApi } from "@/lib/api/jobs";

function JobFormPageContent() {
  const searchParams = useSearchParams();
  const editParam = searchParams.get("edit");
  const jobId = editParam ? Number(editParam) : null;
  const isEdit = jobId != null && !Number.isNaN(jobId);

  const { data: job, isLoading, error } = useSWR(
    isEdit ? `/jobs/${jobId}` : null,
    () => jobsApi.get(jobId!),
  );

  if (isEdit && isLoading) {
    return (
      <div className="flex items-center gap-2 text-[var(--muted)] py-20">
        <Loader2 className="animate-spin" size={18} /> Loading job…
      </div>
    );
  }

  if (isEdit && (error || !job)) {
    return <div className="text-red-400 py-20">Job not found</div>;
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-white">
          {isEdit ? "Edit Training Job" : "New Training Job"}
        </h1>
        <p className="text-[var(--muted)] mt-1">
          {isEdit
            ? "Update the configuration and save changes"
            : "Configure and save a new SDXL LoRA training job"}
        </p>
      </div>
      <JobForm
        jobId={isEdit ? jobId! : undefined}
        initialName={job?.name ?? ""}
        initialYaml={job?.config_yaml}
      />
    </div>
  );
}

export default function JobFormPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-[var(--muted)] py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <JobFormPageContent />
    </Suspense>
  );
}
