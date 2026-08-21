import type { StorageEntry } from "@/lib/api/storage";

export interface CatalogItem {
  id: number;
  relative_path: string;
}

export function normalizeRelativePath(path: string): string {
  return path
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
}

export function parentPath(path: string): string {
  const normalized = normalizeRelativePath(path);
  if (!normalized) return "";
  const parts = normalized.split("/");
  parts.pop();
  return parts.join("/");
}

export function childName(path: string): string {
  const normalized = normalizeRelativePath(path);
  if (!normalized) return "";
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? "";
}

export function joinRelativePath(...segments: string[]): string {
  return segments
    .map(normalizeRelativePath)
    .filter(Boolean)
    .join("/");
}

export function isDirectChild(itemPath: string, folderPath: string): boolean {
  const item = normalizeRelativePath(itemPath);
  const folder = normalizeRelativePath(folderPath);

  if (!folder) {
    return !item.includes("/");
  }

  if (!item.startsWith(`${folder}/`)) {
    return false;
  }

  const remainder = item.slice(folder.length + 1);
  return remainder.length > 0 && !remainder.includes("/");
}

export function isCatalogItemAtPath(itemPath: string, folderPath: string): boolean {
  const item = normalizeRelativePath(itemPath);
  const folder = normalizeRelativePath(folderPath);
  if (item === folder) {
    return true;
  }
  return isDirectChild(itemPath, folderPath);
}

export interface PartitionFolderOptions<T extends CatalogItem> {
  entries: StorageEntry[];
  catalogItems: T[];
  currentPath: string;
  excludeFolder?: (entry: StorageEntry, folderPath: string) => boolean;
  onExcludedFolder?: (entry: StorageEntry, folderPath: string) => void;
}

export function partitionFolderContents<T extends CatalogItem>({
  entries,
  catalogItems,
  currentPath,
  excludeFolder,
  onExcludedFolder,
}: PartitionFolderOptions<T>): { folders: StorageEntry[]; items: T[] } {
  const folder = normalizeRelativePath(currentPath);
  const catalogByPath = new Map(
    catalogItems.map((item) => [normalizeRelativePath(item.relative_path), item] as const)
  );

  const items = catalogItems.filter((item) => {
    const itemPath = normalizeRelativePath(item.relative_path);
    if (!folder && !itemPath) {
      return true;
    }
    return isCatalogItemAtPath(item.relative_path, folder);
  });

  const folders = entries.filter((entry) => {
    if (!entry.is_dir) return false;
    const entryPath = normalizeRelativePath(entry.relative_path);
    if (catalogByPath.has(entryPath)) {
      return false;
    }
    if (excludeFolder?.(entry, folder)) {
      onExcludedFolder?.(entry, folder);
      return false;
    }
    return true;
  });

  return { folders, items };
}
