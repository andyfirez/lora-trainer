"use client";

import type { ReactNode } from "react";
import { ArrowLeft, Loader2 } from "lucide-react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import StorageBreadcrumbs from "@/components/storage/StorageBreadcrumbs";
import { parentPath } from "@/lib/storagePaths";

interface StorageBrowserShellProps {
  currentPath: string;
  onNavigate: (path: string) => void;
  root?: string;
  loading?: boolean;
  error?: unknown;
  isEmpty?: boolean;
  emptyHint?: string;
  errorHint?: ReactNode;
  children: ReactNode;
}

export default function StorageBrowserShell({
  currentPath,
  onNavigate,
  root,
  loading = false,
  error,
  isEmpty = false,
  emptyHint,
  errorHint,
  children,
}: StorageBrowserShellProps) {
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

      {root ? <p className="text-xs text-muted break-all">Root: {root}</p> : null}

      {loading ? (
        <div className="flex items-center justify-center gap-2 py-16 text-muted">
          <Loader2 size={18} className="animate-spin" />
          Loading…
        </div>
      ) : error ? (
        <Card className="py-12 text-center text-error space-y-2">
          <p>{error instanceof Error ? error.message : "Failed to browse storage"}</p>
          {errorHint}
        </Card>
      ) : isEmpty ? (
        <Card className="py-16 text-center text-muted space-y-2">
          <p>{currentPath ? "Empty folder" : "Nothing here yet"}</p>
          {emptyHint ? <p className="text-sm">{emptyHint}</p> : null}
        </Card>
      ) : (
        <Card className="p-0 overflow-hidden divide-y divide-border">{children}</Card>
      )}
    </div>
  );
}
