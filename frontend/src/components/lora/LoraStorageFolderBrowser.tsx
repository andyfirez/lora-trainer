"use client";

import useSWR from "swr";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2, Sparkles } from "lucide-react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { lorasApi } from "@/lib/api/loras";
import { storageApi } from "@/lib/api/storage";
import { findLoraByPath, partitionLoraFolderContents } from "@/lib/loraFolderContents";
import { parentPath } from "@/lib/storagePaths";
import StorageBreadcrumbs from "@/components/storage/StorageBreadcrumbs";
import StorageFolderRow from "@/components/storage/StorageFolderRow";
import LoraFolderItem from "@/components/lora/LoraFolderItem";

interface LoraStorageFolderBrowserProps {
  currentPath: string;
  onNavigate: (path: string) => void;
  emptyHint?: string;
}

async function fetchLoraFolder(currentPath: string) {
  const [loras, browse] = await Promise.all([
    lorasApi.list(),
    storageApi.browse("lora", currentPath),
  ]);
  return { loras, browse };
}

export default function LoraStorageFolderBrowser({
  currentPath,
  onNavigate,
  emptyHint,
}: LoraStorageFolderBrowserProps) {
  const router = useRouter();
  const { data, error, isLoading, mutate } = useSWR(
    ["lora-folder", currentPath],
    () => fetchLoraFolder(currentPath),
  );

  const loras = data?.loras ?? [];
  const browse = data?.browse;
  const { folders, loras: visibleLoras, unmatchedLoraDirs } = partitionLoraFolderContents({
    entries: browse?.entries ?? [],
    loras,
    currentPath,
  });

  const openLoraAtPath = async (path: string) => {
    let match = findLoraByPath(loras, path);
    if (!match) {
      const fresh = await fetchLoraFolder(currentPath);
      await mutate(fresh, { revalidate: false });
      match = findLoraByPath(fresh.loras, path);
    }
    if (match) {
      router.push(`/loras/${match.id}`);
    }
  };

  const handleFolderClick = (path: string, isLoraWorkDir: boolean) => {
    if (isLoraWorkDir) {
      void openLoraAtPath(path);
      return;
    }
    onNavigate(path);
  };

  const isEmpty =
    !isLoading && folders.length === 0 && visibleLoras.length === 0 && unmatchedLoraDirs.length === 0;
  const parent = parentPath(currentPath);

  return (
    <div className="space-y-3">
      {currentPath ? (
        <div className="flex items-center gap-3 min-w-0">
          <Button variant="secondary" size="sm" onClick={() => onNavigate(parent)} className="shrink-0">
            <ArrowLeft size={14} />
            Back
          </Button>
          <StorageBreadcrumbs currentPath={currentPath} onNavigate={onNavigate} />
        </div>
      ) : null}

      {browse?.root ? <p className="text-xs text-muted break-all">Root: {browse.root}</p> : null}

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 py-16 text-muted">
          <Loader2 size={18} className="animate-spin" />
          Loading…
        </div>
      ) : error ? (
        <Card className="py-12 text-center text-error space-y-2">
          <p>{error instanceof Error ? error.message : "Failed to browse storage"}</p>
          <p className="text-xs text-muted">Check that the backend is running.</p>
        </Card>
      ) : isEmpty ? (
        <Card className="py-16 text-center text-muted space-y-2">
          <p>{currentPath ? "Empty folder" : "Nothing here yet"}</p>
          {emptyHint ? <p className="text-sm">{emptyHint}</p> : null}
        </Card>
      ) : (
        <Card className="p-0 overflow-hidden divide-y divide-border">
          {folders.map((folder) => (
            <StorageFolderRow
              key={folder.relative_path}
              name={folder.name}
              onClick={() => handleFolderClick(folder.relative_path, folder.is_lora_work_dir ?? false)}
            />
          ))}
          {visibleLoras.map((lora) => (
            <div key={lora.id}>
              <LoraFolderItem lora={lora} />
            </div>
          ))}
          {unmatchedLoraDirs.map((entry) => (
            <button
              key={entry.relative_path}
              type="button"
              onClick={() => void openLoraAtPath(entry.relative_path)}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 text-left transition-colors group"
            >
              <Sparkles size={18} className="text-sampling shrink-0" />
              <span className="flex-1 min-w-0 truncate font-medium text-text group-hover:text-accent transition-colors">
                {entry.name}
              </span>
              <span className="text-xs text-muted shrink-0 hidden sm:inline">Open LoRA</span>
            </button>
          ))}
        </Card>
      )}
    </div>
  );
}
