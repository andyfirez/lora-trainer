"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { settingsApi, type GpuDefaultsInfo } from "@/lib/api/settings";

const MIXED_PRECISION_OPTIONS = [
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

const VAE_DTYPE_OPTIONS = [
  { value: "auto", label: "auto" },
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

const ATTENTION_OPTIONS = [
  { value: "sdpa", label: "SDPA (PyTorch 2.x)" },
  { value: "xformers", label: "xformers" },
  { value: "default", label: "diffusers default" },
];

const labelClassName = "block text-sm font-medium mb-1";
const selectClassName =
  "w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40";

export default function GpuDefaultsSettingsTab() {
  const { data, isLoading, mutate } = useSWR("/settings", () => settingsApi.get());
  const [form, setForm] = useState<GpuDefaultsInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (!data?.gpu_defaults) return;
    setForm(data.gpu_defaults);
  }, [data]);

  const handleSave = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!form) return;
    setError(null);
    setSuccess(false);
    setSaving(true);
    try {
      await settingsApi.patch({
        tf32: form.tf32,
        attention_mechanism: form.attention_mechanism,
        mixed_precision: form.mixed_precision,
        vae_dtype: form.vae_dtype,
        sample_vae_tiling: form.sample_vae_tiling,
      });
      await mutate();
      setSuccess(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save GPU settings");
    } finally {
      setSaving(false);
    }
  };

  if (isLoading || !data || !form) {
    return <div className="text-muted">Loading…</div>;
  }

  return (
    <Card className="max-w-xl space-y-4">
      <p className="text-sm text-muted">
        Global GPU defaults for training and sampling. TF32 and attention apply to all jobs.
        Other fields are defaults — configs can override them per job.
      </p>
      <form onSubmit={handleSave} className="space-y-4">
        <div>
          <label className={labelClassName}>Attention mechanism</label>
          <select
            className={selectClassName}
            value={form.attention_mechanism}
            onChange={(e) =>
              setForm({ ...form, attention_mechanism: e.target.value as GpuDefaultsInfo["attention_mechanism"] })
            }
          >
            {ATTENTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClassName}>Mixed precision (default)</label>
          <select
            className={selectClassName}
            value={form.mixed_precision}
            onChange={(e) =>
              setForm({ ...form, mixed_precision: e.target.value as GpuDefaultsInfo["mixed_precision"] })
            }
          >
            {MIXED_PRECISION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClassName}>VAE dtype (default)</label>
          <select
            className={selectClassName}
            value={form.vae_dtype}
            onChange={(e) =>
              setForm({ ...form, vae_dtype: e.target.value as GpuDefaultsInfo["vae_dtype"] })
            }
          >
            {VAE_DTYPE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.tf32}
            onChange={(e) => setForm({ ...form, tf32: e.target.checked })}
          />
          <span className="text-sm">TF32 matmul (Ampere+ GPUs)</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.sample_vae_tiling}
            onChange={(e) => setForm({ ...form, sample_vae_tiling: e.target.checked })}
          />
          <span className="text-sm">VAE tiling (default for sampling)</span>
        </label>
        {error && <p className="text-sm text-error">{error}</p>}
        {success && <p className="text-sm text-success">Settings saved.</p>}
        <Button type="submit" disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </form>
    </Card>
  );
}
