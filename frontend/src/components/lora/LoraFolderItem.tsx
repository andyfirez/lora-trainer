"use client";

import Link from "next/link";
import { Sparkles } from "lucide-react";
import type { TrainedLora } from "@/types";
import StorageCatalogRow from "@/components/storage/StorageCatalogRow";

interface LoraFolderItemProps {
  lora: TrainedLora;
}

function LoraMeta({ lora }: { lora: TrainedLora }) {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
      <span className="truncate max-w-xs">{lora.base_model_name}</span>
      <span>{new Date(lora.created_at).toLocaleDateString()}</span>
      {lora.job_id != null ? (
        <Link href={`/jobs/${lora.job_id}`} className="text-accent hover:underline">
          Job #{lora.job_id}
        </Link>
      ) : null}
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
