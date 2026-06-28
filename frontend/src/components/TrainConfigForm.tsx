"use client";

import useSWR from "swr";
import Link from "next/link";
import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import SampleSamplerFields from "@/components/SampleSamplerFields";
import { datasetsApi } from "@/lib/api/datasets";
import {
  applyOptimizerPreset,
  optimizerOptions,
  type OptimizerType,
} from "@/lib/optimizerPresets";
import type { Dataset } from "@/types";

type Config = Record<string, any>;

interface TrainConfigFormProps {
  config: Config;
  onChange: (config: Config) => void;
}

const inputClass =
  "w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-1.5 text-sm text-white placeholder-[var(--muted)] focus:outline-none focus:border-[var(--accent)]";
const selectClass =
  "w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-1.5 text-sm text-white focus:outline-none focus:border-[var(--accent)]";
const labelClass = "block text-xs font-medium text-[var(--muted)] mb-1";
const sectionClass = "bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-white mb-3";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      {children}
    </div>
  );
}

function TextInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <Field label={label}>
      <input
        type="text"
        className={inputClass}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </Field>
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
        className={inputClass}
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
      <select className={selectClass} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
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
        className="w-4 h-4 rounded accent-[var(--accent)]"
        checked={!!checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-sm text-white">{label}</span>
    </label>
  );
}

const weightDtypeOptions = [
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

const lrSchedulerOptions = [
  { value: "constant", label: "Constant" },
  { value: "constant_with_warmup", label: "Constant with Warmup" },
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
  { value: "cosine_with_restarts", label: "Cosine with Restarts" },
  { value: "polynomial", label: "Polynomial" },
];

export default function TrainConfigForm({ config, onChange }: TrainConfigFormProps) {
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }

  function setNested(parent: string, key: string, value: unknown) {
    onChange({
      ...config,
      [parent]: { ...(config[parent] ?? {}), [key]: value },
    });
  }

  function setOptimizerType(type: OptimizerType) {
    onChange(applyOptimizerPreset(config, type));
  }

  const optimizerType: OptimizerType = config.optimizer?.type ?? "adamw_8bit";
  const isAdamFamily = optimizerType === "adamw" || optimizerType === "adamw_8bit";
  const isAdafactor = optimizerType === "adafactor";
  const isProdigy = optimizerType === "prodigy";

  const concepts: Config[] = config.concepts ?? [];
  const samplePrompts: string[] = config.sample_prompts ?? [];
  const { data: datasets, isLoading: datasetsLoading } = useSWR("/datasets", () => datasetsApi.list());

  const datasetOptions = (datasets ?? []).map((d: Dataset) => ({
    value: String(d.id),
    label: d.name,
  }));

  function datasetById(id: number | undefined): Dataset | undefined {
    if (id == null) return undefined;
    return datasets?.find((d) => d.id === id);
  }

  function updateConcept(i: number, key: string, value: unknown) {
    const next = concepts.map((c, idx) => (idx === i ? { ...c, [key]: value } : c));
    set("concepts", next);
  }

  function addConcept() {
    const defaultDatasetId = datasets?.[0]?.id;
    if (defaultDatasetId == null) return;
    set("concepts", [
      ...concepts,
      { dataset_id: defaultDatasetId, caption_extension: ".txt", repeats: 1 },
    ]);
  }

  function removeConcept(i: number) {
    set("concepts", concepts.filter((_, idx) => idx !== i));
  }

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
      {/* Model */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Model</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PathInput
            label="Base Model"
            value={config.base_model_name ?? ""}
            onChange={(v) => set("base_model_name", v)}
            placeholder="stabilityai/stable-diffusion-xl-base-1.0 or D:\models\sdxl"
            pickerTitle="Select Base Model"
            kind="model"
          />
          <PathInput
            label="Output Folder"
            value={config.output_dir ?? ""}
            onChange={(v) => set("output_dir", v)}
            placeholder="D:\loras\output"
            pickerTitle="Select Output Folder"
            kind="directory"
          />
        </div>
        <TextInput
          label="LoRA Name"
          value={config.lora_name ?? ""}
          onChange={(v) => set("lora_name", v)}
          placeholder="my_lora"
        />
        <SelectInput
          label="Output Format"
          value={config.output_format ?? "safetensors"}
          onChange={(v) => set("output_format", v)}
          options={[
            { value: "safetensors", label: "safetensors" },
            { value: "pt", label: "pt" },
          ]}
        />
      </section>

      {/* LoRA */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>LoRA</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <NumberInput
            label="Rank"
            value={config.lora_rank}
            onChange={(v) => set("lora_rank", v)}
            min={1}
            max={256}
            placeholder="16"
          />
          <NumberInput
            label="Alpha"
            value={config.lora_alpha}
            onChange={(v) => set("lora_alpha", v)}
            min={0}
            step={0.1}
            placeholder="16.0"
          />
          <NumberInput
            label="Dropout"
            value={config.lora_dropout}
            onChange={(v) => set("lora_dropout", v)}
            min={0}
            max={0.999}
            step={0.01}
            placeholder="0.0"
          />
        </div>
      </section>

      {/* Training Targets */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Training Targets</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--muted)]">
                <th className="pb-2 font-medium">Component</th>
                <th className="pb-2 font-medium">Train</th>
                <th className="pb-2 font-medium">Weight Dtype</th>
              </tr>
            </thead>
            <tbody className="space-y-2">
              {(["unet", "text_encoder_1", "text_encoder_2"] as const).map((part) => (
                <tr key={part} className="border-t border-[var(--border)]">
                  <td className="py-2 pr-4 text-white font-mono text-xs">{part}</td>
                  <td className="py-2 pr-4">
                    <input
                      type="checkbox"
                      className="w-4 h-4 accent-[var(--accent)]"
                      checked={!!(config[part]?.train ?? (part === "unet"))}
                      onChange={(e) => setNested(part, "train", e.target.checked)}
                    />
                  </td>
                  <td className="py-2">
                    <select
                      className="rounded-lg bg-[var(--bg)] border border-[var(--border)] px-2 py-1 text-xs text-white focus:outline-none focus:border-[var(--accent)]"
                      value={config[part]?.weight_dtype ?? "float16"}
                      onChange={(e) => setNested(part, "weight_dtype", e.target.value)}
                    >
                      {weightDtypeOptions.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Training Hyperparameters */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Training</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <NumberInput label="Epochs" value={config.epochs} onChange={(v) => set("epochs", v)} min={1} placeholder="10" />
          <NumberInput label="Batch Size" value={config.batch_size} onChange={(v) => set("batch_size", v)} min={1} placeholder="1" />
          <NumberInput
            label="Grad Accumulation Steps"
            value={config.gradient_accumulation_steps}
            onChange={(v) => set("gradient_accumulation_steps", v)}
            min={1}
            placeholder="1"
          />
          <NumberInput
            label="Learning Rate"
            value={config.learning_rate}
            onChange={(v) => set("learning_rate", v)}
            min={0}
            step={0.0001}
            placeholder="0.0001"
          />
          <SelectInput
            label="LR Scheduler"
            value={config.lr_scheduler ?? "constant"}
            onChange={(v) => set("lr_scheduler", v)}
            options={lrSchedulerOptions}
          />
          <NumberInput
            label="LR Warmup Steps"
            value={config.lr_warmup_steps}
            onChange={(v) => set("lr_warmup_steps", v)}
            min={0}
            placeholder="10"
          />
        </div>
      </section>

      {/* Optimizer */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Optimizer</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SelectInput
            label="Type"
            value={optimizerType}
            onChange={(v) => setOptimizerType(v as OptimizerType)}
            options={[...optimizerOptions]}
          />
          {(isAdamFamily || isProdigy) && (
            <>
              <NumberInput
                label="Weight Decay"
                value={config.optimizer?.weight_decay}
                onChange={(v) => setNested("optimizer", "weight_decay", v)}
                min={0}
                step={0.01}
                placeholder="0.01"
              />
              <NumberInput
                label="Beta 1"
                value={config.optimizer?.beta1}
                onChange={(v) => setNested("optimizer", "beta1", v)}
                min={0}
                max={0.999}
                step={0.01}
                placeholder="0.9"
              />
              <NumberInput
                label="Beta 2"
                value={config.optimizer?.beta2}
                onChange={(v) => setNested("optimizer", "beta2", v)}
                min={0}
                max={0.999}
                step={0.001}
                placeholder="0.999"
              />
            </>
          )}
          {isAdafactor && (
            <>
              <CheckboxInput
                label="Relative Step"
                checked={config.optimizer?.relative_step ?? false}
                onChange={(v) => setNested("optimizer", "relative_step", v)}
              />
              <CheckboxInput
                label="Scale Parameter"
                checked={config.optimizer?.scale_parameter ?? false}
                onChange={(v) => setNested("optimizer", "scale_parameter", v)}
              />
              <CheckboxInput
                label="Warmup Init"
                checked={config.optimizer?.warmup_init ?? false}
                onChange={(v) => setNested("optimizer", "warmup_init", v)}
              />
            </>
          )}
          {isProdigy && (
            <>
              <CheckboxInput
                label="Decouple"
                checked={config.optimizer?.decouple ?? true}
                onChange={(v) => setNested("optimizer", "decouple", v)}
              />
              <CheckboxInput
                label="Use Bias Correction"
                checked={config.optimizer?.use_bias_correction ?? true}
                onChange={(v) => setNested("optimizer", "use_bias_correction", v)}
              />
              <CheckboxInput
                label="Safeguard Warmup"
                checked={config.optimizer?.safeguard_warmup ?? true}
                onChange={(v) => setNested("optimizer", "safeguard_warmup", v)}
              />
              <NumberInput
                label="d0"
                value={config.optimizer?.d0}
                onChange={(v) => setNested("optimizer", "d0", v)}
                min={0}
                step={0.00001}
                placeholder="0.00001"
              />
              <NumberInput
                label="d Coef"
                value={config.optimizer?.d_coef}
                onChange={(v) => setNested("optimizer", "d_coef", v)}
                min={0}
                step={0.1}
                placeholder="1.0"
              />
            </>
          )}
        </div>
      </section>

      {/* Data / Concepts */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Data</div>
        <NumberInput
          label="Resolution"
          value={config.resolution}
          onChange={(v) => set("resolution", v)}
          min={64}
          max={2048}
          step={64}
          placeholder="1024"
        />
        <div className="space-y-3 mt-2">
          <div className="text-xs font-medium text-[var(--muted)]">Concepts</div>
          {datasetsLoading ? (
            <div className="text-sm text-[var(--muted)]">Loading datasets…</div>
          ) : !datasets?.length ? (
            <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center space-y-3">
              <p className="text-sm text-[var(--muted)]">
                No datasets yet. Create a dataset to specify training data.
              </p>
              <Link
                href="/datasets"
                className="inline-flex items-center gap-1.5 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium"
              >
                Create Dataset
              </Link>
            </div>
          ) : (
            <>
              {concepts.map((concept, i) => {
                const selectedDataset = datasetById(concept.dataset_id);
                return (
                  <div key={i} className="relative rounded-lg border border-[var(--border)] p-4 bg-[var(--bg)]">
                    <button
                      type="button"
                      onClick={() => removeConcept(i)}
                      className="absolute top-2 right-2 p-1 rounded hover:bg-white/10 text-[var(--muted)] hover:text-red-400"
                    >
                      <X size={13} />
                    </button>
                    <div className="text-xs text-[var(--muted)] mb-3">Concept {i + 1}</div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="md:col-span-1">
                        <SelectInput
                          label="Dataset"
                          value={concept.dataset_id != null ? String(concept.dataset_id) : ""}
                          onChange={(v) => updateConcept(i, "dataset_id", Number(v))}
                          options={datasetOptions}
                        />
                        {selectedDataset && (
                          <p className="text-xs text-[var(--muted)] mt-1 break-all">{selectedDataset.image_dir}</p>
                        )}
                        {concept.dataset_id != null && !selectedDataset && (
                          <p className="text-xs text-red-400 mt-1">Dataset not found</p>
                        )}
                      </div>
                      <Field label="Caption Extension">
                        <input
                          type="text"
                          className={inputClass}
                          value={concept.caption_extension ?? ".txt"}
                          onChange={(e) => updateConcept(i, "caption_extension", e.target.value)}
                          placeholder=".txt"
                        />
                      </Field>
                      <Field label="Repeats">
                        <input
                          type="number"
                          className={inputClass}
                          value={concept.repeats ?? 1}
                          min={1}
                          onChange={(e) => updateConcept(i, "repeats", Number(e.target.value))}
                        />
                      </Field>
                    </div>
                  </div>
                );
              })}
              <button
                type="button"
                onClick={addConcept}
                className="flex items-center gap-1.5 text-sm text-[var(--muted)] hover:text-white border border-dashed border-[var(--border)] hover:border-white/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
              >
                <Plus size={13} /> Add Concept
              </button>
            </>
          )}
        </div>
      </section>

      {/* Optimization */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Optimization</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <SelectInput
            label="Mixed Precision"
            value={config.mixed_precision ?? "bfloat16"}
            onChange={(v) => set("mixed_precision", v)}
            options={weightDtypeOptions}
          />
          <NumberInput
            label="Seed (optional)"
            value={config.seed ?? null}
            onChange={(v) => set("seed", v)}
            min={0}
            placeholder="random"
          />
          <div className="flex items-center pb-1">
            <CheckboxInput
              label="Gradient Checkpointing"
              checked={config.gradient_checkpointing ?? true}
              onChange={(v) => set("gradient_checkpointing", v)}
            />
          </div>
        </div>
      </section>

      {/* Performance */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Performance</div>

        {/* Caching */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-[var(--muted)]">Caching</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
            <CheckboxInput
              label="Cache Latents (RAM)"
              checked={config.cache_latents ?? true}
              onChange={(v) => set("cache_latents", v)}
            />
            <CheckboxInput
              label="Cache Latents to Disk (.npz)"
              checked={config.cache_latents_to_disk ?? false}
              onChange={(v) => set("cache_latents_to_disk", v)}
            />
            <CheckboxInput
              label="Cache Text Encoder Outputs (RAM)"
              checked={config.cache_text_encoder_outputs ?? true}
              onChange={(v) => set("cache_text_encoder_outputs", v)}
            />
            <CheckboxInput
              label="Cache Text Encoder Outputs to Disk"
              checked={config.cache_text_encoder_outputs_to_disk ?? false}
              onChange={(v) => set("cache_text_encoder_outputs_to_disk", v)}
            />
          </div>
          {(config.cache_text_encoder_outputs ?? true) && (config.unet?.train === false) && (config.text_encoder_1?.train || config.text_encoder_2?.train) && (
            <p className="text-xs text-red-400 mt-1">
              Cache Text Encoder Outputs is incompatible with training text encoders. Disable one of them.
            </p>
          )}
        </div>

        {/* Attention & Precision */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end mt-4">
          <SelectInput
            label="Attention Mechanism"
            value={config.attention_mechanism ?? "sdpa"}
            onChange={(v) => set("attention_mechanism", v)}
            options={[
              { value: "sdpa", label: "SDPA (default, PyTorch 2.x)" },
              { value: "xformers", label: "xformers" },
              { value: "default", label: "diffusers default" },
            ]}
          />
          <div className="flex flex-col gap-2 pb-1">
            <CheckboxInput
              label="TF32 (Ampere+ GPUs)"
              checked={config.tf32 ?? true}
              onChange={(v) => set("tf32", v)}
            />
            <CheckboxInput
              label="torch.compile (slower start)"
              checked={config.torch_compile ?? false}
              onChange={(v) => set("torch_compile", v)}
            />
          </div>
        </div>

        {/* DataLoader */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <NumberInput
            label="DataLoader Workers (0 = main thread)"
            value={config.num_dataloader_workers ?? 0}
            onChange={(v) => set("num_dataloader_workers", v ?? 0)}
            min={0}
            placeholder="0"
          />
          <div className="flex items-end pb-1">
            <CheckboxInput
              label="Pin Memory (requires workers > 0)"
              checked={config.dataloader_pin_memory ?? true}
              onChange={(v) => set("dataloader_pin_memory", v)}
            />
          </div>
        </div>
      </section>

      {/* Checkpointing & Sampling */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Checkpointing & Sampling</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NumberInput
            label="Save Every N Epochs"
            value={config.save_every_n_epochs}
            onChange={(v) => set("save_every_n_epochs", v)}
            min={1}
            placeholder="1"
          />
          <div className="flex items-center pb-1">
            <CheckboxInput
              label="Run sampling after training for intermediate checkpoints"
              checked={config.sample_after_training ?? false}
              onChange={(v) => set("sample_after_training", v)}
            />
          </div>
          <NumberInput
            label="Sample Steps"
            value={config.sample_steps ?? 30}
            onChange={(v) => set("sample_steps", v)}
            min={1}
            placeholder="30"
          />
          <NumberInput
            label="Sample CFG Scale"
            value={config.sample_cfg_scale ?? 7.5}
            onChange={(v) => set("sample_cfg_scale", v)}
            min={0}
            step={0.5}
            placeholder="7.5"
          />
          <NumberInput
            label="Sample Width (optional)"
            value={config.sample_width ?? null}
            onChange={(v) => set("sample_width", v)}
            min={64}
            max={2048}
            step={64}
            placeholder="same as resolution"
          />
          <NumberInput
            label="Sample Height (optional)"
            value={config.sample_height ?? null}
            onChange={(v) => set("sample_height", v)}
            min={64}
            max={2048}
            step={64}
            placeholder="same as resolution"
          />
          <SampleSamplerFields
            sampleScheduler={config.sample_scheduler ?? "euler"}
            onChange={set}
          />
        </div>
        <div className="mt-3">
          <TextInput
            label="Negative Prompt"
            value={config.sample_negative_prompt ?? ""}
            onChange={(v) => set("sample_negative_prompt", v)}
            placeholder="low quality, blurry, ..."
          />
        </div>
        <div className="space-y-2 mt-2">
          <div className="text-xs font-medium text-[var(--muted)]">Sample Prompts</div>
          {samplePrompts.map((prompt, i) => (
            <div key={i} className="flex items-center gap-2">
              <input
                type="text"
                className={inputClass}
                value={prompt}
                onChange={(e) => updatePrompt(i, e.target.value)}
                placeholder={`Prompt ${i + 1}`}
              />
              <button
                type="button"
                onClick={() => removePrompt(i)}
                className="p-1.5 rounded hover:bg-white/10 text-[var(--muted)] hover:text-red-400 shrink-0"
              >
                <X size={13} />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addPrompt}
            className="flex items-center gap-1.5 text-sm text-[var(--muted)] hover:text-white border border-dashed border-[var(--border)] hover:border-white/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
          >
            <Plus size={13} /> Add Prompt
          </button>
        </div>
      </section>
    </div>
  );
}
