"use client";

import { ChevronRight } from "lucide-react";
import { normalizeRelativePath } from "@/lib/storagePaths";

interface StorageBreadcrumbsProps {
  currentPath: string;
  onNavigate: (path: string) => void;
  /** Label for the root segment in browse modals. Omit on list pages (breadcrumbs hidden at root). */
  rootLabel?: string;
}

export default function StorageBreadcrumbs({ currentPath, onNavigate, rootLabel }: StorageBreadcrumbsProps) {
  const normalized = normalizeRelativePath(currentPath);
  const parts = normalized ? normalized.split("/") : [];

  if (!rootLabel && parts.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-1 text-sm text-muted min-w-0 flex-wrap">
      {rootLabel ? (
        <button type="button" className="hover:text-text shrink-0" onClick={() => onNavigate("")}>
          {rootLabel}
        </button>
      ) : null}
      {parts.map((part, index) => {
        const path = parts.slice(0, index + 1).join("/");
        return (
          <span key={path} className="flex items-center gap-1 min-w-0">
            <ChevronRight size={12} className="shrink-0" />
            <button type="button" className="hover:text-text truncate" onClick={() => onNavigate(path)}>
              {part}
            </button>
          </span>
        );
      })}
    </div>
  );
}
