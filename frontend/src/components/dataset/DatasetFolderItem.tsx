"use client";

import useSWR from "swr";
import { Database, Image as ImageIcon, Trash2 } from "lucide-react";
import { datasetsApi } from "@/lib/api/datasets";
import type { Dataset } from "@/types";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import StorageCatalogRow from "@/components/storage/StorageCatalogRow";

interface DatasetFolderItemProps {
  dataset: Dataset;
  onDelete: () => void;
}

export default function DatasetFolderItem({ dataset, onDelete }: DatasetFolderItemProps) {
  const { data: images } = useSWR(`/datasets/${dataset.id}/images`, () => datasetsApi.listImages(dataset.id));

  const meta = (
    <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
      <span className="inline-flex items-center gap-1">
        <ImageIcon size={12} />
        {images?.images.length ?? "…"} images
      </span>
      {dataset.target_resolution != null ? <Badge>{dataset.target_resolution}px</Badge> : null}
      <Badge variant={dataset.preprocess_ready ? "success" : "warning"}>
        {dataset.preprocess_ready ? "Ready" : "Not prepared"}
      </Badge>
    </span>
  );

  return (
    <div>
      <StorageCatalogRow
        href={`/datasets/${dataset.id}`}
        icon={<Database size={18} className="text-accent" />}
        title={dataset.name}
        meta={meta}
        actions={
          <Button
            variant="ghost"
            size="icon"
            onClick={(event) => {
              event.preventDefault();
              onDelete();
            }}
            className="text-error hover:text-error"
            aria-label="Delete dataset"
          >
            <Trash2 size={14} />
          </Button>
        }
      />
      <div className="px-4 pb-3 sm:hidden">{meta}</div>
    </div>
  );
}
