"use client";

import useSWR from "swr";
import { ArrowLeft, Loader2 } from "lucide-react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { storageApi, type StorageKind } from "@/lib/api/storage";
import { parentPath, partitionFolderContents } from "@/lib/storagePaths";
import StorageBreadcrumbs from "@/components/storage/StorageBreadcrumbs";
import StorageFolderRow from "@/components/storage/StorageFolderRow";

interface StorageFolderBrowserProps<T extends { id: number; relative_path: string }> {
  kind: StorageKind;
  items: T[];
  currentPath: string;
  onNavigate: (path: string) => void;
  renderItem: (item: T) => React.ReactNode;
  catalogLoading?: boolean;
  emptyHint?: string;
}

export default function StorageFolderBrowser<T extends { id: number; relative_path: string }>({
  kind,
  items,
  currentPath,
  onNavigate,
  renderItem,
  catalogLoading = false,
  emptyHint,
}: StorageFolderBrowserProps<T>) {
  const browseKey = `/storage/browse?kind=${kind}&path=${currentPath}`;
  const { data: browse, isLoading: browseLoading, error } = useSWR(browseKey, () =>
    storageApi.browse(kind, currentPath)
  );

  const loading = catalogLoading || browseLoading;
  const { folders, items: catalogItems } = partitionFolderContents({
    entries: browse?.entries ?? [],
    catalogItems: items,
    currentPath,
  });

  const isEmpty = !loading && folders.length === 0 && catalogItems.length === 0;
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

      {loading ? (
        <div className="flex items-center justify-center gap-2 py-16 text-muted">
          <Loader2 size={18} className="animate-spin" />
          Loading…
        </div>
      ) : error ? (
        <Card className="py-12 text-center text-error">
          {error instanceof Error ? error.message : "Failed to browse storage"}
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
              onClick={() => onNavigate(folder.relative_path)}
            />
          ))}
          {catalogItems.map((item) => (
            <div key={item.id}>{renderItem(item)}</div>
          ))}
        </Card>
      )}
    </div>
  );
}
