"use client";

import dynamic from "next/dynamic";
import { Download } from "lucide-react";
import Button from "@/components/ui/Button";
import { downloadTextFile } from "@/lib/download";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

export interface YamlViewerProps {
  value: string;
  height?: number;
  downloadFilename?: string;
  className?: string;
}

export default function YamlViewer({ value, height = 400, downloadFilename, className }: YamlViewerProps) {
  const handleDownload = () => {
    if (!downloadFilename) return;
    downloadTextFile(value, downloadFilename, "text/yaml");
  };

  return (
    <div className={className}>
      {downloadFilename && (
        <div className="flex justify-end mb-2">
          <Button variant="secondary" size="sm" onClick={handleDownload}>
            <Download size={14} /> Download YAML
          </Button>
        </div>
      )}
      <div className="overflow-hidden rounded-lg border border-border" style={{ height }}>
        <MonacoEditor
          height={`${height}px`}
          defaultLanguage="yaml"
          theme="vs-dark"
          value={value}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
        />
      </div>
    </div>
  );
}
