"use client";

import { useMemo } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Download } from "lucide-react";
import { parse as yamlParse } from "yaml";
import Button from "@/components/ui/Button";
import Card, { CardTitle } from "@/components/ui/Card";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

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

  const handleDownloadYaml = () => {
    if (!configYaml) return;
    const blob = new Blob([configYaml], { type: "text/yaml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.yaml`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

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
      <div className="flex flex-wrap items-center justify-between gap-3">
        <CardTitle className="text-base">Config</CardTitle>
        <Button variant="secondary" size="sm" onClick={handleDownloadYaml}>
          <Download size={14} /> Download YAML
        </Button>
      </div>

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

      <div className="overflow-hidden rounded-lg border border-border" style={{ height: 400 }}>
        <MonacoEditor
          height="400px"
          defaultLanguage="yaml"
          theme="vs-dark"
          value={configYaml}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
        />
      </div>
    </Card>
  );
}
