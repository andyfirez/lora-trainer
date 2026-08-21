"use client";

import { useMemo } from "react";
import Link from "next/link";
import { parse as yamlParse } from "yaml";
import Card, { CardTitle } from "@/components/ui/Card";
import YamlViewer from "@/components/YamlViewer";

interface LoraConfigPanelProps {
  name: string;
  configYaml: string | null;
}

function extractDatasetIds(configYaml: string): number[] {
  try {
    const parsed = yamlParse(configYaml) as { concepts?: Array<{ dataset_id?: number }> };
    const concepts = parsed?.concepts ?? [];
    const ids = concepts
      .map((c) => c.dataset_id)
      .filter((id): id is number => typeof id === "number" && Number.isFinite(id));
    return [...new Set(ids)];
  } catch {
    return [];
  }
}

export default function LoraConfigPanel({ name, configYaml }: LoraConfigPanelProps) {
  const datasetIds = useMemo(
    () => (configYaml ? extractDatasetIds(configYaml) : []),
    [configYaml],
  );

  if (!configYaml) {
    return (
      <Card>
        <CardTitle className="text-base mb-2">Config</CardTitle>
        <p className="text-sm text-muted">No config snapshot — this LoRA was imported from disk.</p>
      </Card>
    );
  }

  return (
    <Card className="space-y-4">
      <CardTitle className="text-base">Config</CardTitle>

      {datasetIds.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs text-muted">Linked datasets</div>
          <ul className="flex flex-wrap gap-2">
            {datasetIds.map((id) => (
              <li key={id}>
                <Link
                  href={`/datasets/${id}`}
                  className="text-sm text-accent hover:underline"
                >
                  Dataset #{id}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}

      <YamlViewer value={configYaml} downloadFilename={`${name}.yaml`} />
    </Card>
  );
}
