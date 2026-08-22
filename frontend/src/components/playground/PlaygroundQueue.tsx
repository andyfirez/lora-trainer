"use client";

import { X } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import type { QueueEntry } from "@/components/playground/queueTypes";
import { cancelActionLabel } from "@/lib/runnableStatus";

interface PlaygroundQueueProps {
  items: QueueEntry[];
  onCancel: (item: QueueEntry) => void;
}

export default function PlaygroundQueue({ items, onCancel }: PlaygroundQueueProps) {
  if (!items.length) return null;
  return (
    <div className="px-3 py-2">
      <div className="mb-1 text-xs font-medium text-muted">Queue</div>
      <ul className="space-y-1">
        {items.map((item) => (
          <li key={`${item.kind}-${item.id}`} className="flex items-center gap-2 text-xs">
            <StatusBadge status={item.status} />
            <span className="min-w-0 flex-1 truncate text-text" title={item.label}>
              {item.kind === "lora" ? "Training · " : ""}
              {item.label}
            </span>
            {item.kind === "sampling" || item.status === "queued" ? (
              <button
                type="button"
                title={cancelActionLabel(item.status)}
                onClick={() => onCancel(item)}
                className="p-1 rounded text-error hover:bg-white/10"
              >
                <X size={12} />
              </button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
