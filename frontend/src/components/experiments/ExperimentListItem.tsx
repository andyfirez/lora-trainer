"use client";

import Link from "next/link";
import { ChevronRight } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import { countCombinations, experimentSummary, getParameters } from "@/lib/experimentUtils";
import type { SamplingResponse } from "@/types";

interface ExperimentListItemProps {
  experiment: SamplingResponse;
}

function promptPreview(experiment: SamplingResponse): string {
  const prompt = getParameters(experiment.config).prompt;
  const value = prompt?.mode === "vary" ? prompt.values?.[0] : prompt?.value;
  return String(value ?? "").trim() || experiment.name || "Experiment";
}

export default function ExperimentListItem({ experiment }: ExperimentListItemProps) {
  const parameters = getParameters(experiment.config);
  const imageCount = countCombinations(parameters);
  const preview = promptPreview(experiment);

  return (
    <Link
      href={`/experiments/${experiment.id}`}
      className="flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-3 transition-colors hover:border-accent/40 hover:bg-white/[0.02]"
    >
      <StatusBadge status={experiment.status} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-text">{experiment.name}</p>
        <p className="truncate text-xs text-muted" title={preview}>
          {preview}
        </p>
        <p className="text-xs text-muted">
          {imageCount} image{imageCount === 1 ? "" : "s"} · {experimentSummary(parameters)}
        </p>
      </div>
      <ChevronRight size={16} className="shrink-0 text-muted" />
    </Link>
  );
}
