"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import Input from "@/components/ui/Input";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { settingsApi } from "@/lib/api/settings";

export default function StorageSettingsTab() {
  const { data, isLoading, mutate } = useSWR("/settings", () => settingsApi.get());
  const [datasetsRoot, setDatasetsRoot] = useState("");
  const [baseModelsRoot, setBaseModelsRoot] = useState("");
  const [loraRoot, setLoraRoot] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (!data?.storage) return;
    setDatasetsRoot(data.storage.datasets_root);
    setBaseModelsRoot(data.storage.base_models_root);
    setLoraRoot(data.storage.lora_root);
  }, [data]);

  const handleSave = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(false);
    setSaving(true);
    try {
      await settingsApi.patch({
        datasets_root: datasetsRoot,
        base_models_root: baseModelsRoot,
        lora_root: loraRoot,
      });
      await mutate();
      setSuccess(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save storage settings");
    } finally {
      setSaving(false);
    }
  };

  if (isLoading || !data) {
    return <div className="text-muted">Loading…</div>;
  }

  return (
    <Card className="max-w-xl space-y-4">
      <p className="text-sm text-muted">
        Managed storage roots are saved to config.toml. Relative paths in the database are resolved against these folders at runtime.
      </p>
      <form onSubmit={handleSave} className="space-y-4">
        <Input label="Datasets root" value={datasetsRoot} onChange={(e) => setDatasetsRoot(e.target.value)} />
        <Input label="Base models root" value={baseModelsRoot} onChange={(e) => setBaseModelsRoot(e.target.value)} />
        <Input label="LoRA root" value={loraRoot} onChange={(e) => setLoraRoot(e.target.value)} />
        {error && <p className="text-sm text-error">{error}</p>}
        {success && <p className="text-sm text-success">Storage settings saved.</p>}
        <Button type="submit" disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </form>
    </Card>
  );
}
