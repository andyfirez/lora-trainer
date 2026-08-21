import type { StorageEntry } from "./api/storage.ts";
import type { LoraResponse } from "../types/index.ts";
import { isCatalogItemAtPath, normalizeRelativePath, partitionFolderContents } from "./storagePaths.ts";

export interface UnmatchedLoraDir {
  name: string;
  relative_path: string;
}

export function findLoraByPath(loras: LoraResponse[], path: string): LoraResponse | undefined {
  const normalized = normalizeRelativePath(path);
  return loras.find((lora) => normalizeRelativePath(lora.relative_path) === normalized);
}

export function partitionLoraFolderContents({
  entries,
  loras,
  currentPath,
}: {
  entries: StorageEntry[];
  loras: LoraResponse[];
  currentPath: string;
}): { folders: StorageEntry[]; loras: LoraResponse[]; unmatchedLoraDirs: UnmatchedLoraDir[] } {
  const unmatchedLoraDirs: UnmatchedLoraDir[] = [];

  const { folders, items } = partitionFolderContents({
    entries,
    catalogItems: loras,
    currentPath,
    excludeFolder: (entry) => entry.is_lora_work_dir === true,
    onExcludedFolder: (entry, folderPath) => {
      if (isCatalogItemAtPath(entry.relative_path, folderPath)) {
        unmatchedLoraDirs.push({ name: entry.name, relative_path: entry.relative_path });
      }
    },
  });

  return { folders, loras: items, unmatchedLoraDirs };
}
