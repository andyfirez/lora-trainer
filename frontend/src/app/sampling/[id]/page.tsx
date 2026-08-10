"use client";

import { use } from "react";
import useSWR from "swr";
import Link from "next/link";
import dynamic from "next/dynamic";
import { ArrowLeft, Download, Loader2, Play, Square } from "lucide-react";
import { samplingsApi } from "@/lib/api/samplings";
import StatusBadge from "@/components/StatusBadge";
import SamplingRunPanel from "@/components/sampling/SamplingRunPanel";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  params: Promise<{ id: string }>;
}

export default function SamplingDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const id = Number(idParam);
  const { data: sampling, isLoading, mutate } = useSWR(`/samplings/${id}`, () => samplingsApi.get(id), {
    refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 2000),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }
  if (!sampling) return <div className="text-error">Sampling not found</div>;

  const handleEnqueue = async () => {
    await samplingsApi.enqueue(id);
    mutate();
  };
  const handleCancel = async () => {
    await samplingsApi.cancel(id);
    mutate();
  };
  const handleDownloadYaml = () => {
    const blob = new Blob([sampling.config_yaml], { type: "text/yaml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${sampling.name}.yaml`;
    a.click();
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link
          href="/sampling"
          className="p-2 rounded-lg hover:bg-white/5 text-muted hover:text-text"
          aria-label="Back to sampling"
        >
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-text font-display">{sampling.name}</h1>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <StatusBadge status={sampling.status} />
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap shrink-0">
          {(sampling.status === "draft" ||
            sampling.status === "failed" ||
            sampling.status === "cancelled" ||
            sampling.status === "orphan") && (
            <Button variant="success" size="sm" onClick={() => void handleEnqueue()}>
              <Play size={13} /> Enqueue
            </Button>
          )}
          {(sampling.status === "queued" || sampling.status === "running") && (
            <Button variant="danger" size="sm" onClick={() => void handleCancel()}>
              <Square size={13} /> {sampling.status === "running" ? "Stop" : "Cancel"}
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={handleDownloadYaml}>
            <Download size={13} /> YAML
          </Button>
        </div>
      </div>

      <SamplingRunPanel sampling={sampling} />

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-muted">Config YAML</h2>
        <Card padding="none" className="overflow-hidden" style={{ height: 400 }}>
          <MonacoEditor
            height="100%"
            language="yaml"
            theme="vs-dark"
            value={sampling.config_yaml}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, scrollBeyondLastLine: false }}
          />
        </Card>
      </div>
    </div>
  );
}
