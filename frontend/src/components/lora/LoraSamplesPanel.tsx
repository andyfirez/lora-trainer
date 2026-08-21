"use client";

import useSWR from "swr";
import { lorasApi } from "@/lib/api/loras";
import { mediaUrl } from "@/lib/media";
import Card, { CardTitle } from "@/components/ui/Card";

interface LoraSamplesPanelProps {
  loraId: number;
  status: string;
}

export default function LoraSamplesPanel({ loraId, status }: LoraSamplesPanelProps) {
  const { data, isLoading } = useSWR(
    status === "completed" ? `/loras/${loraId}/samples` : null,
    () => lorasApi.getSamples(loraId),
  );

  if (status !== "completed") return null;

  const samples = data?.samples.filter((s) => s.kind === "legacy" || s.kind == null) ?? [];

  if (isLoading) {
    return (
      <Card>
        <CardTitle className="text-base mb-2">Samples</CardTitle>
        <p className="text-sm text-muted">Loading samples…</p>
      </Card>
    );
  }

  if (!samples.length) return null;

  return (
    <Card className="space-y-4">
      <CardTitle className="text-base">Samples</CardTitle>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {samples.map((sample) => (
          <a
            key={sample.path}
            href={mediaUrl(sample.url)}
            target="_blank"
            rel="noreferrer"
            className="block"
          >
            <img
              src={mediaUrl(sample.url)}
              alt={sample.filename}
              className="rounded-lg border border-border object-cover aspect-square w-full"
            />
          </a>
        ))}
      </div>
    </Card>
  );
}
