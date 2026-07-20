"use client";

import { useState } from "react";
import useSWR from "swr";
import { jobsApi } from "@/lib/api/jobs";
import type { SweepManifestResponse } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface SweepGridViewerProps {
  jobId: number;
  status: string;
}

export default function SweepGridViewer({ jobId, status }: SweepGridViewerProps) {
  const [gridIndex, setGridIndex] = useState(0);
  const [selectedCell, setSelectedCell] = useState<number | null>(null);

  const { data: manifest } = useSWR(
    status === "completed" ? `/jobs/${jobId}/sweep-manifest` : null,
    () => jobsApi.getSweepManifest(jobId),
  );

  if (!manifest?.grids.length && !manifest?.images.length) {
    return null;
  }

  const grids = manifest.grids;
  const currentGrid = grids[gridIndex];

  return (
    <div className="space-y-6">
      {grids.length > 0 && currentGrid && (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-sm font-medium text-muted">
              Grid {gridIndex + 1}/{grids.length}
              {currentGrid.title ? ` — ${currentGrid.title}` : ""}
            </h2>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={gridIndex === 0}
                onClick={() => setGridIndex((i) => i - 1)}
                className="px-3 py-1 text-sm rounded-lg border border-border disabled:opacity-40"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={gridIndex >= grids.length - 1}
                onClick={() => setGridIndex((i) => i + 1)}
                className="px-3 py-1 text-sm rounded-lg border border-border disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
          <a href={`${API_BASE_URL}${currentGrid.url}`} target="_blank" rel="noreferrer">
            <img
              src={`${API_BASE_URL}${currentGrid.url}`}
              alt={currentGrid.title || `Grid ${gridIndex + 1}`}
              className="rounded-xl border border-border w-full"
            />
          </a>
          <div className="text-xs text-muted">
            X: {currentGrid.x.param} · Y: {currentGrid.y.param}
          </div>
        </div>
      )}

      {manifest.images.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-muted">All cells ({manifest.total_images})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {manifest.images.map((img) => (
              <button
                key={img.index}
                type="button"
                onClick={() => setSelectedCell(selectedCell === img.index ? null : img.index)}
                className={`text-left rounded-lg border overflow-hidden transition-colors ${
                  selectedCell === img.index ? "border-accent" : "border-border"
                }`}
              >
                <img
                  src={`${API_BASE_URL}${img.url}`}
                  alt={`Cell ${img.index}`}
                  className="object-cover aspect-square w-full"
                />
                {selectedCell === img.index && (
                  <div className="p-2 text-xs text-muted space-y-0.5">
                    {Object.entries(img.params).map(([k, v]) => (
                      <div key={k}>
                        <span className="text-text">{k}:</span> {String(v).slice(0, 40)}
                      </div>
                    ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
