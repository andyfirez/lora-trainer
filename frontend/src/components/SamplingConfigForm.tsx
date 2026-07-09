"use client";

import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import SampleSamplerFields from "@/components/SampleSamplerFields";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";

type Config = Record<string, unknown>;

interface SamplingConfigFormProps {
  config: Config;
  onChange: (config: Config) => void;
}

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className={labelClassName}>{label}</label>
      {children}
    </div>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        className={inputClassName}
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(raw === "" ? null : Number(raw));
        }}
      />
    </Field>
  );
}

function SelectInput({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Field label={label}>
      <select className={selectClassName} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

function CheckboxInput({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <input
        type="checkbox"
        className="w-4 h-4 rounded accent-accent"
        checked={!!checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-sm text-text">{label}</span>
    </label>
  );
}

export default function SamplingConfigForm({ config, onChange }: SamplingConfigFormProps) {
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  const samplePrompts: string[] = (config.sample_prompts as string[]) ?? [];

  function updatePrompt(i: number, value: string) {
    const next = samplePrompts.map((p, idx) => (idx === i ? value : p));
    set("sample_prompts", next);
  }

  function addPrompt() {
    set("sample_prompts", [...samplePrompts, ""]);
  }

  function removePrompt(i: number) {
    set("sample_prompts", samplePrompts.filter((_, idx) => idx !== i));
  }

  return (
    <div className="space-y-5">
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Model & Output</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PathInput
            label="Base Model"
            value={(config.base_model_name as string) ?? ""}
            onChange={(v) => set("base_model_name", v)}
            placeholder="stabilityai/stable-diffusion-xl-base-1.0"
            pickerTitle="Select Base Model"
            kind="model"
          />
          <PathInput
            label="Output Folder"
            value={(config.output_dir as string) ?? ""}
            onChange={(v) => set("output_dir", v)}
            placeholder="D:\loras\output"
            pickerTitle="Select Output Folder"
            kind="directory"
          />
        </div>
      </section>

      <section className={sectionClass}>
        <div className={sectionTitleClass}>Sampling</div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Sample Steps">
            <input
              type="number"
              className={inputClassName}
              value={(config.sample_steps as number) ?? 30}
              min={1}
              onChange={(e) => set("sample_steps", Number(e.target.value))}
            />
          </Field>
          <Field label="CFG Scale">
            <input
              type="number"
              className={inputClassName}
              value={(config.sample_cfg_scale as number) ?? 7.5}
              min={0}
              step={0.5}
              onChange={(e) => set("sample_cfg_scale", Number(e.target.value))}
            />
          </Field>
          <NumberInput
            label="Sample Width (optional)"
            value={(config.sample_width as number | null | undefined) ?? null}
            onChange={(v) => set("sample_width", v)}
            min={64}
            max={2048}
            step={64}
            placeholder="1024"
          />
          <NumberInput
            label="Sample Height (optional)"
            value={(config.sample_height as number | null | undefined) ?? null}
            onChange={(v) => set("sample_height", v)}
            min={64}
            max={2048}
            step={64}
            placeholder="1024"
          />
          <SampleSamplerFields
            sampleScheduler={(config.sample_scheduler as string) ?? "euler"}
            onChange={set}
          />
          <SelectInput
            label="Attention"
            value={(config.attention_mechanism as string) ?? "sdpa"}
            onChange={(v) => set("attention_mechanism", v)}
            options={[
              { value: "xformers", label: "xformers" },
              { value: "sdpa", label: "SDPA (PyTorch 2.x)" },
              { value: "default", label: "Default" },
            ]}
          />
          <SelectInput
            label="Mixed Precision"
            value={(config.mixed_precision as string) ?? "float16"}
            onChange={(v) => set("mixed_precision", v)}
            options={[
              { value: "bfloat16", label: "bfloat16" },
              { value: "float16", label: "float16" },
              { value: "float32", label: "float32" },
            ]}
          />
          <SelectInput
            label="VAE Dtype"
            value={(config.vae_dtype as string) ?? "auto"}
            onChange={(v) => set("vae_dtype", v)}
            options={[
              { value: "auto", label: "Auto (bf16 on Ampere+)" },
              { value: "bfloat16", label: "bfloat16" },
              { value: "float16", label: "float16" },
              { value: "float32", label: "float32" },
            ]}
          />
        </div>
        <CheckboxInput
          label="TF32 matmul"
          checked={(config.tf32 as boolean) ?? true}
          onChange={(v) => set("tf32", v)}
        />
        <Field label="Negative Prompt">
          <input
            type="text"
            className={inputClassName}
            value={(config.sample_negative_prompt as string) ?? ""}
            onChange={(e) => set("sample_negative_prompt", e.target.value)}
            placeholder="low quality, blurry, ..."
          />
        </Field>
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted">Prompts</div>
          {samplePrompts.map((prompt, i) => (
            <div key={i} className="flex items-center gap-2">
              <input
                type="text"
                className={inputClassName}
                value={prompt}
                onChange={(e) => updatePrompt(i, e.target.value)}
                placeholder={`Prompt ${i + 1}`}
              />
              <button
                type="button"
                onClick={() => removePrompt(i)}
                className="p-1.5 rounded hover:bg-white/10 text-muted hover:text-error shrink-0"
              >
                <X size={13} />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addPrompt}
            className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border hover:border-text/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
          >
            <Plus size={13} /> Add Prompt
          </button>
        </div>
      </section>
    </div>
  );
}
