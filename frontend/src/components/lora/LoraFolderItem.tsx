"use client";

import { Sparkles } from "lucide-react";
import type { LoraResponse } from "@/types";
import StatusBadge from "@/components/StatusBadge";
import StorageCatalogRow from "@/components/storage/StorageCatalogRow";

interface LoraFolderItemProps {
  lora: LoraResponse;
}

function LoraMeta({ lora }: { lora: LoraResponse }) {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
      <StatusBadge status={lora.status} />
      {lora.status === "queued" && lora.queue_position != null && (
        <span>Queue #{lora.queue_position}</span>
      )}
      <span className="truncate max-w-xs">{lora.base_model_name}</span>
      <span>{new Date(lora.created_at).toLocaleDateString()}</span>
      {lora.status === "running" && lora.progress_step != null && lora.progress_total != null && (
        <span>
          step {lora.progress_step}/{lora.progress_total}
        </span>
      )}
    </span>
  );
}

export default function LoraFolderItem({ lora }: LoraFolderItemProps) {
  const meta = <LoraMeta lora={lora} />;

  return (
    <div>
      <StorageCatalogRow
        href={`/loras/${lora.id}`}
        icon={<Sparkles size={18} className="text-sampling" />}
        title={lora.name}
        meta={meta}
      />
      <div className="px-4 pb-3 sm:hidden">{meta}</div>
    </div>
  );
}
