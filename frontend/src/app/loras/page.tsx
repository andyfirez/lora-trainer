"use client";

import useSWR from "swr";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback } from "react";
import { lorasApi } from "@/lib/api/loras";
import StorageFolderBrowser from "@/components/storage/StorageFolderBrowser";
import LoraFolderItem from "@/components/lora/LoraFolderItem";
import PageHeader from "@/components/ui/PageHeader";
import { normalizeRelativePath } from "@/lib/storagePaths";

function LorasPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentPath = normalizeRelativePath(searchParams.get("path") ?? "");
  const { data: loras, isLoading } = useSWR("/loras", () => lorasApi.list());

  const navigateToPath = useCallback(
    (path: string, replace = false) => {
      const normalized = normalizeRelativePath(path);
      const params = new URLSearchParams(searchParams.toString());
      if (normalized) {
        params.set("path", normalized);
      } else {
        params.delete("path");
      }
      const query = params.toString();
      const href = query ? `/loras?${query}` : "/loras";
      if (replace) {
        router.replace(href);
      } else {
        router.push(href);
      }
    },
    [router, searchParams]
  );

  const handleNavigate = useCallback(
    (path: string) => {
      const normalized = normalizeRelativePath(path);
      const current = normalizeRelativePath(currentPath);
      const isBack =
        normalized === "" ||
        (current.startsWith(`${normalized}/`) && normalized.split("/").length < current.split("/").length);
      navigateToPath(path, isBack);
    },
    [currentPath, navigateToPath]
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="LoRAs"
        description="Successfully trained LoRA models with frozen configs and artifacts"
      />

      <StorageFolderBrowser
        kind="lora"
        items={loras ?? []}
        currentPath={currentPath}
        onNavigate={handleNavigate}
        catalogLoading={isLoading}
        emptyHint="Complete a training job or place LoRA weights under the LoRA root to auto-discover."
        renderItem={(lora) => <LoraFolderItem lora={lora} />}
      />
    </div>
  );
}

export default function LorasPage() {
  return (
    <Suspense fallback={<div className="text-muted py-20">Loading…</div>}>
      <LorasPageContent />
    </Suspense>
  );
}
