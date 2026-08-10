import type { StorageEntry } from "./api/storage.ts";
import type { LoraResponse } from "../types/index.ts";
import { isCatalogItemAtPath, normalizeRelativePath } from "./storagePaths.ts";

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
  const folder = normalizeRelativePath(currentPath);
  const catalogByPath = new Map(
    loras.map((lora) => [normalizeRelativePath(lora.relative_path), lora] as const),
  );

  const visibleLoras = loras.filter((lora) => {
    const itemPath = normalizeRelativePath(lora.relative_path);
    if (!folder && !itemPath) {
      return true;
    }
    return isCatalogItemAtPath(lora.relative_path, folder);
  });

  const unmatchedLoraDirs: UnmatchedLoraDir[] = [];

  const folders = entries.filter((entry) => {
    if (!entry.is_dir) {
      return false;
    }
    const entryPath = normalizeRelativePath(entry.relative_path);
    if (catalogByPath.has(entryPath)) {
      return false;
    }
    if (entry.is_lora_work_dir) {
      if (isCatalogItemAtPath(entry.relative_path, folder)) {
        unmatchedLoraDirs.push({ name: entry.name, relative_path: entry.relative_path });
      }
      return false;
    }
    return true;
  });

  return { folders, loras: visibleLoras, unmatchedLoraDirs };
}
