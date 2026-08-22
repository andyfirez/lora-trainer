"use client";

import { useState } from "react";
import { ArrowLeftRight, Dices, Undo2 } from "lucide-react";
import Slider from "@/components/ui/Slider";
import { InheritedCheckboxField, InheritedSelectField } from "@/components/ui/InheritedGpuField";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { formSectionTitleClass } from "@/components/ui/FormSection";
import type { GpuDefaultsInfo } from "@/lib/api/settings";
import { MIXED_PRECISION_OPTIONS, VAE_DTYPE_OPTIONS } from "@/lib/gpuConfigUtils";
import { diffusersSchedulerOptions } from "@/lib/sampleSamplerOptions";
import { getParameters, setParameter } from "@/lib/sweepUtils";

const DEFAULT_WIDTH = 832;
const DEFAULT_HEIGHT = 1216;

interface PlaygroundCompactSettingsProps {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
  batchCount: number;
  onBatchCountChange: (value: number) => void;
  gpuDefaults?: GpuDefaultsInfo;
  lastSeed: number | null;
}

function fixedNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export default function PlaygroundCompactSettings({
  config,
  onChange,
  batchCount,
  onBatchCountChange,
  gpuDefaults,
  lastSeed,
}: PlaygroundCompactSettingsProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const parameters = getParameters(config);
  const seedParam = parameters.seed ?? { mode: "fixed" as const, value: null };
  const width = fixedNumber(parameters.width?.value, DEFAULT_WIDTH);
  const height = fixedNumber(parameters.height?.value, DEFAULT_HEIGHT);
  const steps = fixedNumber(parameters.steps?.value, 30);
  const cfg = fixedNumber(parameters.cfg_scale?.value, 7.5);
  const scheduler = String(parameters.scheduler?.value ?? "euler");

  function setFixed(key: "steps" | "cfg_scale" | "width" | "height" | "scheduler" | "seed", value: unknown) {
    onChange(setParameter(config, key, { mode: "fixed", value }));
  }

  function set(key: string, value: unknown) {
    const next: Record<string, unknown> = { ...config };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  }

  return (
    <div className="space-y-3 p-3">
      <div>
        <label className={labelClassName}>Sampling method</label>
        <select
          className={selectClassName}
          value={scheduler}
          onChange={(event) => setFixed("scheduler", event.target.value)}
        >
          {diffusersSchedulerOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <Slider label="Sampling steps" value={steps} min={1} max={150} step={1} onChange={(value) => setFixed("steps", value)} />

      <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-2">
        <Slider label="Width" value={width} min={64} max={2048} step={8} onChange={(value) => setFixed("width", value)} />
        <button
          type="button"
          title="Swap width/height"
          onClick={() => {
            let next = setParameter(config, "width", { mode: "fixed", value: height });
            next = setParameter(next, "height", { mode: "fixed", value: width });
            onChange(next);
          }}
          className="mb-1 rounded-lg border border-border p-2 text-muted hover:bg-white/5 hover:text-text"
        >
          <ArrowLeftRight size={14} />
        </button>
        <Slider label="Height" value={height} min={64} max={2048} step={8} onChange={(value) => setFixed("height", value)} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Slider
          label="Batch count"
          value={batchCount}
          min={1}
          max={32}
          step={1}
          onChange={onBatchCountChange}
        />
        <Slider
          label="CFG Scale"
          value={cfg}
          min={0}
          max={30}
          step={0.1}
          onChange={(value) => setFixed("cfg_scale", value)}
        />
      </div>

      <div className="space-y-1">
        <label className={labelClassName}>Seed</label>
        <div className="flex items-center gap-2">
          <input
            type="number"
            className={inputClassName}
            value={seedParam.value == null ? "" : String(seedParam.value)}
            placeholder="random"
            onChange={(event) =>
              setFixed("seed", event.target.value === "" ? null : Number(event.target.value))
            }
          />
          <button
            type="button"
            title="Random seed"
            onClick={() => setFixed("seed", null)}
            className="rounded-lg border border-border p-2 text-muted hover:bg-white/5 hover:text-text"
          >
            <Dices size={15} />
          </button>
          <button
            type="button"
            title="Reuse last seed"
            disabled={lastSeed == null}
            onClick={() => lastSeed != null && setFixed("seed", lastSeed)}
            className="rounded-lg border border-border p-2 text-muted hover:bg-white/5 hover:text-text disabled:opacity-40"
          >
            <Undo2 size={15} />
          </button>
        </div>
      </div>

      <p className="text-xs text-muted">
        {Math.max(1, batchCount)} image{batchCount === 1 ? "" : "s"} this Generate
      </p>

      <section className="border-t border-border pt-3">
        <button
          type="button"
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className={`${formSectionTitleClass} mb-0 w-full text-left flex items-center justify-between`}
        >
          Advanced (performance)
          <span className="text-muted text-xs">{advancedOpen ? "▲" : "▼"}</span>
        </button>
        {advancedOpen && (
          <div className="space-y-4">
            {gpuDefaults ? (
              <>
                <InheritedSelectField
                  label="Mixed Precision"
                  value={config.mixed_precision as string | undefined}
                  globalDefault={gpuDefaults.mixed_precision}
                  options={MIXED_PRECISION_OPTIONS}
                  onChange={(v) => set("mixed_precision", v)}
                  paramKey="mixed_precision"
                />
                <InheritedSelectField
                  label="VAE Dtype"
                  value={config.vae_dtype as string | undefined}
                  globalDefault={gpuDefaults.vae_dtype}
                  options={VAE_DTYPE_OPTIONS}
                  onChange={(v) => set("vae_dtype", v)}
                  paramKey="vae_dtype"
                />
                <InheritedCheckboxField
                  label="VAE tiling"
                  value={config.sample_vae_tiling as boolean | undefined}
                  globalDefault={gpuDefaults.sample_vae_tiling}
                  onChange={(v) => set("sample_vae_tiling", v)}
                  paramKey="sample_vae_tiling"
                />
              </>
            ) : (
              <p className="text-sm text-muted">Loading GPU defaults…</p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
