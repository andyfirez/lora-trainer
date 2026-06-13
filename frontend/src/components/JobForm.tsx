"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { Save } from "lucide-react";
import { parse as yamlParse, stringify as yamlStringify } from "yaml";
import { jobsApi } from "@/lib/api/jobs";
import { TrainConfig } from "@/lib/defaultConfig";
import TrainConfigForm from "@/components/TrainConfigForm";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface JobFormProps {
  initialName?: string;
  initialYaml?: string;
  jobId?: number;
}

type Tab = "form" | "yaml";

export default function JobForm({ initialName = "", initialYaml = TrainConfig.DEFAULT_YAML, jobId }: JobFormProps) {
  const router = useRouter();
  const [name, setName] = useState(initialName);
  const [yaml, setYaml] = useState(initialYaml);
  const [tab, setTab] = useState<Tab>("form");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const config = useMemo(() => {
    try {
      return yamlParse(yaml) ?? {};
    } catch {
      return {};
    }
  }, [yaml]);

  function handleConfigChange(newConfig: Record<string, any>) {
    try {
      setYaml(yamlStringify(newConfig));
    } catch {
      // ignore serialization errors
    }
  }

  function handleYamlChange(value: string) {
    setYaml(value);
  }

  const handleSave = async () => {
    if (!name.trim()) { setError("Name is required"); return; }
    setSaving(true);
    setError(null);
    try {
      if (jobId) {
        await jobsApi.update(jobId, { name, config_yaml: yaml });
      } else {
        const job = await jobsApi.create(name, yaml);
        router.push(`/jobs/${job.id}`);
        return;
      }
      router.push(`/jobs/${jobId}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setYaml((ev.target?.result as string) || "");
    reader.readAsText(file);
  };

  return (
    <div className="flex flex-col gap-4">
      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 text-sm">{error}</div>
      )}

      {/* Header row: name + import + save */}
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-[var(--muted)] mb-1">Job Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-sdxl-lora"
            className="w-full rounded-lg bg-[var(--surface)] border border-[var(--border)] px-3 py-2 text-white placeholder-[var(--muted)] focus:outline-none focus:border-[var(--accent)]"
          />
        </div>
        <div className="flex items-end gap-2 pb-0.5">
          <label className="cursor-pointer text-sm text-[var(--muted)] hover:text-white transition-colors border border-[var(--border)] rounded-lg px-3 py-2">
            Import YAML
            <input type="file" accept=".yaml,.yml" className="hidden" onChange={handleImport} />
          </label>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            <Save size={14} />
            {saving ? "Saving…" : "Save Job"}
          </button>
        </div>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {(["form", "yaml"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              tab === t
                ? "text-white border border-b-[var(--bg)] border-[var(--border)] bg-[var(--bg)] -mb-px"
                : "text-[var(--muted)] hover:text-white"
            }`}
          >
            {t === "form" ? "Form" : "YAML"}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "form" ? (
        <TrainConfigForm config={config} onChange={handleConfigChange} />
      ) : (
        <div className="rounded-xl overflow-hidden border border-[var(--border)]" style={{ height: 520 }}>
          <MonacoEditor
            height="100%"
            language="yaml"
            theme="vs-dark"
            value={yaml}
            onChange={(v) => handleYamlChange(v ?? "")}
            options={{ minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
          />
        </div>
      )}
    </div>
  );
}
