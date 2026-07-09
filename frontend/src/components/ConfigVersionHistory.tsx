"use client";

import { useState } from "react";
import useSWR from "swr";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { configsApi } from "@/lib/api/configs";
import type { JobConfigVersionSummary } from "@/types";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface ConfigVersionHistoryProps {
  configId: number;
  activeVersion: number | null;
}

export default function ConfigVersionHistory({ configId, activeVersion }: ConfigVersionHistoryProps) {
  const { data: versions, isLoading } = useSWR(
    `/configs/${configId}/versions`,
    () => configsApi.listVersions(configId),
  );
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [previewYaml, setPreviewYaml] = useState<string | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  const handleSelectVersion = async (version: number) => {
    if (selectedVersion === version) {
      setSelectedVersion(null);
      setPreviewYaml(null);
      return;
    }
    setSelectedVersion(version);
    setLoadingPreview(true);
    try {
      const entry = await configsApi.getVersion(configId, version);
      setPreviewYaml(entry.config_yaml);
    } finally {
      setLoadingPreview(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted text-sm py-4">
        <Loader2 className="animate-spin" size={16} /> Loading version history…
      </div>
    );
  }

  if (!versions?.length) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-text font-display">Version History</h2>
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="bg-surface">
            <tr>
              <th className="px-4 py-3 text-left text-muted font-medium">Version</th>
              <th className="px-4 py-3 text-left text-muted font-medium">LoRA Name</th>
              <th className="px-4 py-3 text-left text-muted font-medium">Output Files</th>
              <th className="px-4 py-3 text-left text-muted font-medium">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {versions.map((entry: JobConfigVersionSummary) => (
              <tr
                key={entry.version}
                onClick={() => void handleSelectVersion(entry.version)}
                className={`cursor-pointer hover:bg-white/[0.02] transition-colors ${
                  selectedVersion === entry.version ? "bg-white/[0.04]" : ""
                }`}
              >
                <td className="px-4 py-3 text-text">
                  v{entry.version}
                  {entry.version === activeVersion && (
                    <span className="ml-2 text-xs rounded-full bg-accent-muted text-accent px-2 py-0.5">
                      active
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-muted">{entry.lora_name ?? "—"}</td>
                <td className="px-4 py-3 text-muted">
                  {entry.lora_name ? `${entry.lora_name}_v${entry.version}` : "—"}
                </td>
                <td className="px-4 py-3 text-muted">
                  {new Date(entry.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selectedVersion !== null && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-text">Version {selectedVersion} YAML</h3>
          {loadingPreview ? (
            <div className="flex items-center gap-2 text-muted text-sm py-8 justify-center">
              <Loader2 className="animate-spin" size={16} /> Loading…
            </div>
          ) : (
            <div className="rounded-xl overflow-hidden border border-border" style={{ height: 360 }}>
              <MonacoEditor
                height="100%"
                language="yaml"
                theme="vs-dark"
                value={previewYaml ?? ""}
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  fontSize: 13,
                  scrollBeyondLastLine: false,
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
