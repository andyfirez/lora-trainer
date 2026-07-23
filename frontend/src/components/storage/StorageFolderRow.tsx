"use client";

import { ChevronRight, Folder } from "lucide-react";

interface StorageFolderRowProps {
  name: string;
  onClick: () => void;
}

export default function StorageFolderRow({ name, onClick }: StorageFolderRowProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 text-left transition-colors"
    >
      <Folder size={18} className="text-accent shrink-0" />
      <span className="flex-1 min-w-0 truncate text-text">{name}</span>
      <ChevronRight size={16} className="text-muted shrink-0" />
    </button>
  );
}
