"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { parse as yamlParse, stringify as yamlStringify } from "yaml";
import { settingsApi } from "@/lib/api/settings";
import { applySparseGpuOverrides } from "@/lib/gpuConfigUtils";

export type YamlConfigTab = "form" | "yaml";

export interface UseYamlConfigFormOptions {
  defaultYaml: string;
  redirectTo: (id: number) => string;
  create: (params: { name: string; configYaml: string; enqueue: boolean }) => Promise<{ id: number }>;
  validate?: (config: Record<string, unknown>) => string | null;
}

export function useYamlConfigForm({ defaultYaml, redirectTo, create, validate }: UseYamlConfigFormOptions) {
  const router = useRouter();
  const [name, setName] = useState("");
  const [yaml, setYaml] = useState(defaultYaml);
  const [tab, setTab] = useState<YamlConfigTab>("form");
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
    const validationError = validate?.(config as Record<string, unknown>);
    if (validationError) {
      setError(validationError);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const created = await create({ name: name.trim(), configYaml: yaml, enqueue });
      router.push(redirectTo(created.id));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  };

  return {
    name,
    setName,
    yaml,
    setYaml,
    tab,
    setTab,
    enqueue,
    setEnqueue,
    saving,
    error,
    gpuDefaults,
    config: config as Record<string, unknown>,
    handleConfigChange,
    handleImport,
    handleCreate,
  };
}
