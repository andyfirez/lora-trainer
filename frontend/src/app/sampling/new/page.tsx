"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import useSWR from "swr";
import { Loader2, Play, Save } from "lucide-react";
import { parse as yamlParse, stringify as yamlStringify } from "yaml";
import { inputClassName } from "@/components/ui/Input";
import Checkbox from "@/components/ui/Checkbox";
import PageHeader from "@/components/ui/PageHeader";
import Tabs from "@/components/ui/Tabs";
import { samplingsApi } from "@/lib/api/samplings";
import { settingsApi } from "@/lib/api/settings";
import { applySparseGpuOverrides } from "@/lib/gpuConfigUtils";
import { SamplingConfig } from "@/lib/defaultConfig";
import SamplingConfigForm from "@/components/SamplingConfigForm";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

type Tab = "form" | "yaml";

function NewSamplingPageContent() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [yaml, setYaml] = useState(SamplingConfig.DEFAULT_YAML);
  const [tab, setTab] = useState<Tab>("form");
  const [enqueue, setEnqueue] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: settingsData } = useSWR("/settings", () => settingsApi.get());
  const gpuDefaults = settingsData?.gpu_defaults;

  const config = useMemo(() => {
    try {
      return yamlParse(yaml) ?? {};
    } catch {
      return {};
    }
  }, [yaml]);

  function handleConfigChange(newConfig: Record<string, unknown>) {
    try {
      const sparse = gpuDefaults != null ? applySparseGpuOverrides(newConfig, gpuDefaults) : newConfig;
      setYaml(yamlStringify(sparse));
    } catch {
      // ignore serialization errors
    }
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setYaml((ev.target?.result as string) || "");
    reader.readAsText(file);
  };

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      let sampling = await samplingsApi.create({ name: name.trim(), config_yaml: yaml });
      if (enqueue) {
        sampling = await samplingsApi.enqueue(sampling.id);
      }
      router.push(`/sampling/${sampling.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader title="New Sampling" description="Configure and start a new sampling run" />

      <div className="flex flex-col gap-4">
        {error && (
          <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">{error}</div>
        )}

        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-muted mb-1">Sampling Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-sampling-run"
              className={inputClassName}
            />
          </div>
          <label className="cursor-pointer text-sm text-muted hover:text-text transition-colors border border-border rounded-lg px-3 py-2">
            Import YAML
            <input type="file" accept=".yaml,.yml" className="hidden" onChange={handleImport} />
          </label>
        </div>

        <Checkbox label="Enqueue immediately" checked={enqueue} onChange={(e) => setEnqueue(e.target.checked)} />

        <Tabs
          tabs={[
            { value: "form", label: "Form" },
            { value: "yaml", label: "YAML" },
          ]}
          value={tab}
          onChange={setTab}
        />

        {tab === "form" ? (
          <SamplingConfigForm config={config} onChange={handleConfigChange} gpuDefaults={gpuDefaults} />
        ) : (
          <div className="rounded-xl overflow-hidden border border-border" style={{ height: 520 }}>
            <MonacoEditor
              height="100%"
              language="yaml"
              theme="vs-dark"
              value={yaml}
              onChange={(v) => setYaml(v ?? "")}
              options={{ minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
            />
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={() => void handleCreate()}
            disabled={saving}
            className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <Loader2 size={14} className="animate-spin" /> : enqueue ? <Play size={14} /> : <Save size={14} />}
            {saving ? "Creating…" : enqueue ? "Create & Enqueue" : "Create Sampling"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function NewSamplingPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center gap-2 text-muted py-20">
          <Loader2 className="animate-spin" size={18} /> Loading…
        </div>
      }
    >
      <NewSamplingPageContent />
    </Suspense>
  );
}
