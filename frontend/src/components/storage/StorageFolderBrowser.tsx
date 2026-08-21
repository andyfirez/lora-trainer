"use client";

import useSWR from "swr";
import { storageApi, type StorageKind } from "@/lib/api/storage";
import { partitionFolderContents } from "@/lib/storagePaths";
import StorageBrowserShell from "@/components/storage/StorageBrowserShell";
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
    storageApi.browse(kind, currentPath),
  );

  const loading = catalogLoading || browseLoading;
  const { folders, items: catalogItems } = partitionFolderContents({
    entries: browse?.entries ?? [],
    catalogItems: items,
    currentPath,
  });

  const isEmpty = !loading && folders.length === 0 && catalogItems.length === 0;

  return (
    <StorageBrowserShell
      currentPath={currentPath}
      onNavigate={onNavigate}
      root={browse?.root}
      loading={loading}
      error={error}
      isEmpty={isEmpty}
      emptyHint={emptyHint}
    >
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
    </StorageBrowserShell>
  );
}
