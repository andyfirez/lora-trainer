"use client";

import dynamic from "next/dynamic";
import { Loader2, Play, Save } from "lucide-react";
import { inputClassName } from "@/components/ui/Input";
import ErrorAlert from "@/components/ui/ErrorAlert";
import Checkbox from "@/components/ui/Checkbox";
import PageHeader from "@/components/ui/PageHeader";
import Tabs from "@/components/ui/Tabs";
import type { GpuDefaultsInfo } from "@/lib/api/settings";
import type { UseYamlConfigFormOptions, YamlConfigTab } from "@/hooks/useYamlConfigForm";
import { useYamlConfigForm } from "@/hooks/useYamlConfigForm";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

export interface RunnableCreatePageProps extends UseYamlConfigFormOptions {
  title: string;
  description: string;
  nameLabel: string;
  namePlaceholder: string;
  entityLabel: string;
  renderForm: (
    config: Record<string, unknown>,
    onChange: (config: Record<string, unknown>) => void,
    gpuDefaults: GpuDefaultsInfo | undefined,
  ) => React.ReactNode;
}

export default function RunnableCreatePage({
  title,
  description,
  nameLabel,
  namePlaceholder,
  entityLabel,
  renderForm,
  ...formOptions
}: RunnableCreatePageProps) {
  const form = useYamlConfigForm(formOptions);

  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader title={title} description={description} />

      <div className="flex flex-col gap-4">
        {form.error && <ErrorAlert>{form.error}</ErrorAlert>}

        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-muted mb-1">{nameLabel}</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => form.setName(e.target.value)}
              placeholder={namePlaceholder}
              className={inputClassName}
            />
          </div>
          <label className="cursor-pointer text-sm text-muted hover:text-text transition-colors border border-border rounded-lg px-3 py-2">
            Import YAML
            <input type="file" accept=".yaml,.yml" className="hidden" onChange={form.handleImport} />
          </label>
        </div>

        <Checkbox
          label="Enqueue immediately"
          checked={form.enqueue}
          onChange={(e) => form.setEnqueue(e.target.checked)}
        />

        <Tabs
          tabs={[
            { value: "form", label: "Form" },
            { value: "yaml", label: "YAML" },
          ]}
          value={form.tab}
          onChange={(value) => form.setTab(value as YamlConfigTab)}
        />

        {form.tab === "form" ? (
          renderForm(form.config, form.handleConfigChange, form.gpuDefaults)
        ) : (
          <div className="rounded-xl overflow-hidden border border-border" style={{ height: 520 }}>
            <MonacoEditor
              height="100%"
              language="yaml"
              theme="vs-dark"
              value={form.yaml}
              onChange={(v) => form.setYaml(v ?? "")}
              options={{ minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
            />
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={() => void form.handleCreate()}
            disabled={form.saving}
            className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {form.saving ? (
              <Loader2 size={14} className="animate-spin" />
            ) : form.enqueue ? (
              <Play size={14} />
            ) : (
              <Save size={14} />
            )}
            {form.saving ? "Creating…" : form.enqueue ? `Create & Enqueue` : `Create ${entityLabel}`}
          </button>
        </div>
      </div>
    </div>
  );
}
