"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { Save } from "lucide-react";
import { inputClassName } from "@/components/ui/Input";
import { parse as yamlParse, stringify as yamlStringify } from "yaml";
import { trainingsApi } from "@/lib/api/trainings";
import { samplingConfigsApi } from "@/lib/api/samplingConfigs";
import { SamplingConfig, TrainConfig } from "@/lib/defaultConfig";
import TrainConfigForm from "@/components/TrainConfigForm";
import SamplingConfigForm from "@/components/SamplingConfigForm";
import type { ConfigType } from "@/types";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface ConfigFormProps {
  configType: ConfigType;
  initialName?: string;
  initialDescription?: string;
  initialYaml?: string;
  configId?: number;
  onSaved?: () => void;
  saveRedirectBase?: string;
}

type Tab = "form" | "yaml";

export default function ConfigForm({
  configType,
  initialName = "",
  initialDescription = "",
  initialYaml,
  configId,
  onSaved,
  saveRedirectBase,
}: ConfigFormProps) {
  const router = useRouter();
  const defaultYaml = configType === "training" ? TrainConfig.DEFAULT_YAML : SamplingConfig.DEFAULT_YAML;
  const [name, setName] = useState(initialName);
  const [description, setDescription] = useState(initialDescription);
  const [yaml, setYaml] = useState(initialYaml ?? defaultYaml);
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

  function handleConfigChange(newConfig: Record<string, unknown>) {
    try {
      setYaml(yamlStringify(newConfig));
    } catch {
      // ignore serialization errors
    }
  }

  const handleSave = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    if (configType === "training") {
      const parsed = config as Record<string, unknown>;
      const concepts = parsed.concepts;
      if (Array.isArray(concepts)) {
        for (let i = 0; i < concepts.length; i++) {
          const concept = concepts[i];
          if (!concept || typeof concept !== "object" || (concept as Record<string, unknown>).dataset_id == null) {
            setError(`Concept ${i + 1}: select a dataset`);
            return;
          }
        }
      }
    }
    setSaving(true);
    setError(null);
    try {
      if (configId) {
        const updateBody = {
          name,
          config_yaml: yaml,
          description: description || null,
        };
        const updated =
          configType === "training"
            ? await trainingsApi.update(configId, updateBody)
            : await samplingConfigsApi.update(configId, updateBody);
        setYaml(updated.config_yaml);
        onSaved?.();
        router.push(`${saveRedirectBase ?? (configType === "training" ? "/trainings" : "/sampling")}/${configId}`);
        return;
      }
      const createBody = {
        name,
        config_yaml: yaml,
        description: description || null,
      };
      const created =
        configType === "training"
          ? await trainingsApi.create(createBody)
          : await samplingConfigsApi.create(createBody);
      router.push(`${saveRedirectBase ?? (configType === "training" ? "/trainings" : "/sampling")}/${created.id}`);
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
        <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">{error}</div>
      )}

      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-muted mb-1">Config Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={configType === "training" ? "my-sdxl-lora" : "my-sampling-config"}
            className={inputClassName}
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium text-muted mb-1">Description (optional)</label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Notes about this config"
            className={inputClassName}
          />
        </div>
        <div className="flex items-end gap-2 pb-0.5">
          <label className="cursor-pointer text-sm text-muted hover:text-text transition-colors border border-border rounded-lg px-3 py-2">
            Import YAML
            <input type="file" accept=".yaml,.yml" className="hidden" onChange={handleImport} />
          </label>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            <Save size={14} />
            {saving ? "Saving…" : "Save Config"}
          </button>
        </div>
      </div>

      <div className="flex gap-1 border-b border-border">
        {(["form", "yaml"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              tab === t
                ? "text-text border border-b-bg border-border bg-bg -mb-px"
                : "text-muted hover:text-text"
            }`}
          >
            {t === "form" ? "Form" : "YAML"}
          </button>
        ))}
      </div>

      {tab === "form" ? (
        configType === "training" ? (
          <TrainConfigForm config={config} onChange={handleConfigChange} />
        ) : (
          <SamplingConfigForm config={config} onChange={handleConfigChange} />
        )
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
    </div>
  );
}
