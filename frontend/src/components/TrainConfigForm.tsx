"use client";

import { useEffect } from "react";
import useSWR from "swr";
import Link from "next/link";
import { Plus, X } from "lucide-react";
import { parse as yamlParse } from "yaml";
import PathInput from "@/components/PathInput";
import FieldHint from "@/components/FieldHint";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { configsApi } from "@/lib/api/configs";
import { datasetsApi } from "@/lib/api/datasets";
import { trainHint } from "@/lib/trainParameterMetadata";
import {
  applyOptimizerPreset,
  optimizerOptions,
  type OptimizerType,
} from "@/lib/optimizerPresets";
import type { Dataset, JobConfig } from "@/types";

type Config = Record<string, any>;

function stripLoraVersionSuffix(name: string): string {
  return name.replace(/_v\d+$/, "");
}

interface TrainConfigFormProps {
  config: Config;
  onChange: (config: Config) => void;
}

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

function Field({
  label,
  children,
  hint,
  hintAnchor,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
  hintAnchor?: string;
}) {
  return (
    <div>
      <div className="flex items-center mb-1">
        <label className={`${labelClassName} mb-0`}>{label}</label>
        {hint && <FieldHint hint={hint} hintAnchor={hintAnchor} />}
      </div>
      {children}
    </div>
  );
}

function TextInput({
  label,
  value,
  onChange,
  placeholder,
  paramKey,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  paramKey?: string;
}) {
  const hints = paramKey ? trainHint(paramKey) : {};
  return (
    <Field label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
      <input
        type="text"
        className={inputClassName}
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
  disabled,
  paramKey,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
  paramKey?: string;
}) {
  const hints = paramKey ? trainHint(paramKey) : {};
  return (
    <Field label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
      <input
        type="number"
        className={inputClassName}
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
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
  paramKey,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string; disabled?: boolean }[];
  paramKey?: string;
}) {
  const hints = paramKey ? trainHint(paramKey) : {};
  return (
    <Field label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
      <select className={selectClassName} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
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
  disabled,
  paramKey,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  paramKey?: string;
}) {
  const hints = paramKey ? trainHint(paramKey) : {};
  return (
    <label className={`flex items-center gap-2 ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
      <input
        type="checkbox"
        className="w-4 h-4 rounded accent-accent"
        checked={!!checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-sm text-text flex items-center">
        {label}
        {hints.hint && <FieldHint hint={hints.hint} hintAnchor={hints.hintAnchor} />}
      </span>
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

const INLINE_SAMPLING_KEYS = [
  "sample_prompts",
  "sample_negative_prompt",
  "sample_steps",
  "sample_cfg_scale",
  "sample_width",
  "sample_height",
  "sample_scheduler",
  "post_training_sampling_config_id",
  "sample_after_training",
] as const;

const LEGACY_CONCEPT_KEYS = ["image_dir"] as const;

function normalizeConcept(concept: unknown, datasets?: Dataset[]): Config {
  if (!concept || typeof concept !== "object") return concept as Config;
  const raw = concept as Config;
  let datasetId = raw.dataset_id as number | null | undefined;
  const legacyDir = raw.image_dir as string | undefined;

  if (datasetId == null && legacyDir && datasets?.length) {
    datasetId = datasets.find((d) => d.image_dir === legacyDir)?.id;
  }
  if (datasetId == null && datasets?.length) {
    datasetId = datasets[0].id;
  }

  const item = { ...raw };
  if (datasetId != null) {
    item.dataset_id = datasetId;
  }
  if (item.dataset_id != null) {
    for (const key of LEGACY_CONCEPT_KEYS) {
      delete item[key];
    }
  }
  return item;
}

function isTextEncoderTrainingEnabled(config: Config): boolean {
  return Boolean(config.text_encoder_1?.train || config.text_encoder_2?.train);
}

function sanitizeTrainConfig(next: Config, datasets?: Dataset[]): Config {
  let cleaned = stripInlineSamplingFields(next);
  if (cleaned.clip_skip == null) {
    cleaned = { ...cleaned, clip_skip: 2 };
  }
  if (isTextEncoderTrainingEnabled(cleaned)) {
    cleaned = {
      ...cleaned,
      cache_text_encoder_outputs: false,
      cache_text_encoder_outputs_to_disk: false,
    };
  }
  const concepts = cleaned.concepts;
  if (Array.isArray(concepts)) {
    cleaned = {
      ...cleaned,
      concepts: concepts.map((concept) => normalizeConcept(concept, datasets)),
    };
  }
  return cleaned;
}

function stripInlineSamplingFields(next: Config): Config {
  const cleaned = { ...next };
  for (const key of INLINE_SAMPLING_KEYS) {
    delete cleaned[key];
  }
  return cleaned;
}

function samplingPreview(configYaml: string): {
  promptCount: number;
  steps: number;
  scheduler: string;
} | null {
  try {
    const parsed = yamlParse(configYaml) as Record<string, unknown>;
    const prompts = Array.isArray(parsed.sample_prompts) ? parsed.sample_prompts : [];
    return {
      promptCount: prompts.length,
      steps: typeof parsed.sample_steps === "number" ? parsed.sample_steps : 30,
      scheduler: typeof parsed.sample_scheduler === "string" ? parsed.sample_scheduler : "euler",
    };
  } catch {
    return null;
  }
}

export default function TrainConfigForm({ config, onChange }: TrainConfigFormProps) {
  const concepts: Config[] = config.concepts ?? [];
  const { data: datasets, isLoading: datasetsLoading } = useSWR("/datasets", () => datasetsApi.list());
  const { data: samplingConfigs, isLoading: samplingConfigsLoading } = useSWR(
    "/configs/sampling",
    () => configsApi.list("sampling"),
  );

  function emit(next: Config) {
    onChange(sanitizeTrainConfig(next, datasets));
  }

  function set(key: string, value: unknown) {
    let next: Config = { ...config, [key]: value };
    if (key === "checkpointing_enabled" && value === false) {
      next = { ...next, sampling_enabled: false };
    }
    if (key === "cache_latents" && value === false) {
      next = { ...next, cache_latents_to_disk: false };
    }
    if (key === "cache_text_encoder_outputs" && value === false) {
      next = { ...next, cache_text_encoder_outputs_to_disk: false };
    }
    emit(next);
  }

  function setNested(parent: string, key: string, value: unknown) {
    emit({
      ...config,
      [parent]: { ...(config[parent] ?? {}), [key]: value },
    });
  }

  function setOptimizerType(type: OptimizerType) {
    emit(applyOptimizerPreset(config, type));
  }

  const optimizerType: OptimizerType = config.optimizer?.type ?? "adamw_8bit";
  const isAdamFamily = optimizerType === "adamw" || optimizerType === "adamw_8bit";
  const isAdafactor = optimizerType === "adafactor";
  const isProdigy = optimizerType === "prodigy";

  const trainResolution = Number(config.resolution ?? 1024);
  const trainEnableBucket = Boolean(config.enable_bucket);

  function isDatasetCompatible(dataset: Dataset): boolean {
    if (!dataset.preprocess_ready || dataset.target_resolution !== trainResolution) {
      return false;
    }
    return Boolean(dataset.enable_bucket) === trainEnableBucket;
  }

  function datasetOptionLabel(dataset: Dataset): string {
    if (dataset.target_resolution == null) {
      return `${dataset.name} (no target resolution)`;
    }
    if (dataset.target_resolution !== trainResolution) {
      return `${dataset.name} (${dataset.target_resolution}px ≠ ${trainResolution}px)`;
    }
    if (!dataset.preprocess_ready) {
      return `${dataset.name} (not prepared)`;
    }
    if (Boolean(dataset.enable_bucket) !== trainEnableBucket) {
      return `${dataset.name} (bucket ${dataset.enable_bucket ? "on" : "off"} ≠ train)`;
    }
    return dataset.name;
  }

  const datasetOptions = (datasets ?? []).map((d: Dataset) => ({
    value: String(d.id),
    label: datasetOptionLabel(d),
    disabled: !isDatasetCompatible(d),
  }));

  const samplingConfigOptions = (samplingConfigs ?? []).map((c: JobConfig) => ({
    value: String(c.id),
    label: c.name,
  }));

  function datasetById(id: number | undefined): Dataset | undefined {
    if (id == null) return undefined;
    return datasets?.find((d) => d.id === id);
  }

  function samplingConfigById(id: number | undefined): JobConfig | undefined {
    if (id == null) return undefined;
    return samplingConfigs?.find((c) => c.id === id);
  }

  function updateConcept(i: number, key: string, value: unknown) {
    const next = concepts.map((c, idx) => (idx === i ? { ...c, [key]: value } : c));
    set("concepts", next);
  }

  function addConcept() {
    const compatible = datasets?.find((d) => isDatasetCompatible(d));
    const defaultDatasetId = compatible?.id ?? datasets?.[0]?.id;
    if (defaultDatasetId == null) return;
    set("concepts", [
      ...concepts,
      { dataset_id: defaultDatasetId, trigger_words: [], caption_extension: ".txt", repeats: 1 },
    ]);
  }

  function removeConcept(i: number) {
    set("concepts", concepts.filter((_, idx) => idx !== i));
  }

  const selectedSamplingConfig = samplingConfigById(config.sampling_config_id);
  const selectedSamplingPreview = selectedSamplingConfig
    ? samplingPreview(selectedSamplingConfig.config_yaml)
    : null;
  const checkpointingEnabled = config.checkpointing_enabled ?? true;
  const cacheLatentsEnabled = config.cache_latents ?? true;
  const textEncoderTrainingEnabled = isTextEncoderTrainingEnabled(config);
  const cacheTextEncoderEnabled = textEncoderTrainingEnabled
    ? false
    : (config.cache_text_encoder_outputs ?? true);
  const samplingEnabled = config.sampling_enabled ?? false;
  const samplingConfigRequired = samplingEnabled;

  useEffect(() => {
    const normalized = sanitizeTrainConfig(config, datasetsLoading ? undefined : datasets);
    const before = JSON.stringify({
      concepts: config.concepts ?? [],
      clip_skip: config.clip_skip ?? null,
      cache_text_encoder_outputs: config.cache_text_encoder_outputs ?? null,
      cache_text_encoder_outputs_to_disk: config.cache_text_encoder_outputs_to_disk ?? null,
    });
    const after = JSON.stringify({
      concepts: normalized.concepts ?? [],
      clip_skip: normalized.clip_skip ?? null,
      cache_text_encoder_outputs: normalized.cache_text_encoder_outputs ?? null,
      cache_text_encoder_outputs_to_disk: normalized.cache_text_encoder_outputs_to_disk ?? null,
    });
    if (before !== after) {
      onChange(normalized);
    }
  }, [config, datasets, datasetsLoading, onChange]);

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
            {...trainHint("base_model_name")}
          />
          <PathInput
            label="Output Folder"
            value={config.output_dir ?? ""}
            onChange={(v) => set("output_dir", v)}
            placeholder="D:\loras\output"
            pickerTitle="Select Output Folder"
            kind="directory"
            {...trainHint("output_dir")}
          />
        </div>
        <TextInput
          label="LoRA Name"
          value={stripLoraVersionSuffix(config.lora_name ?? "")}
          onChange={(v) => set("lora_name", v)}
          placeholder="my_lora"
          paramKey="lora_name"
        />
        <p className="text-xs text-muted -mt-2">
          Version suffix <code className="text-muted">_vN</code> is added to output files automatically when training starts.
        </p>
        <SelectInput
          label="Output Format"
          value={config.output_format ?? "safetensors"}
          onChange={(v) => set("output_format", v)}
          paramKey="output_format"
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
            placeholder="32"
            paramKey="lora_rank"
          />
          <NumberInput
            label="Alpha"
            value={config.lora_alpha}
            onChange={(v) => set("lora_alpha", v)}
            min={0}
            step={0.1}
            placeholder="32.0"
            paramKey="lora_alpha"
          />
          <NumberInput
            label="Dropout"
            value={config.lora_dropout}
            onChange={(v) => set("lora_dropout", v)}
            min={0}
            max={0.999}
            step={0.01}
            placeholder="0.0"
            paramKey="lora_dropout"
          />
        </div>
      </section>

      {/* Training Targets */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Training Targets</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-2 font-medium">Component</th>
                <th className="pb-2 font-medium">Train</th>
                <th className="pb-2 font-medium">Weight Dtype</th>
                <th className="pb-2 font-medium">Learning Rate</th>
              </tr>
            </thead>
            <tbody className="space-y-2">
              {(["unet", "text_encoder_1", "text_encoder_2"] as const).map((part) => {
                const trainHints = trainHint(`${part}.train`);
                const dtypeHints = trainHint(`${part}.weight_dtype`);
                const lrHints = trainHint(`${part}.learning_rate`);
                const isTraining = !!(config[part]?.train ?? (part === "unet"));
                return (
                <tr key={part} className="border-t border-border">
                  <td className="py-2 pr-4 text-text font-mono text-xs">{part}</td>
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-1">
                      <input
                        type="checkbox"
                        className="w-4 h-4 accent-accent"
                        checked={isTraining}
                        onChange={(e) => setNested(part, "train", e.target.checked)}
                      />
                      {trainHints.hint && (
                        <FieldHint hint={trainHints.hint} hintAnchor={trainHints.hintAnchor} />
                      )}
                    </div>
                  </td>
                  <td className="py-2">
                    <div className="flex items-center gap-1">
                      <select
                        className="rounded-lg bg-bg border border-border px-2 py-1 text-xs text-text focus:outline-none focus:border-accent"
                        value={config[part]?.weight_dtype ?? "float16"}
                        onChange={(e) => setNested(part, "weight_dtype", e.target.value)}
                      >
                        {weightDtypeOptions.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                      {dtypeHints.hint && (
                        <FieldHint hint={dtypeHints.hint} hintAnchor={dtypeHints.hintAnchor} />
                      )}
                    </div>
                  </td>
                  <td className="py-2">
                    {isTraining ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          className="rounded-lg bg-bg border border-border px-2 py-1 text-xs text-text focus:outline-none focus:border-accent w-28"
                          value={config[part]?.learning_rate ?? 0.00005}
                          min={0}
                          step={0.00001}
                          placeholder="0.00005"
                          onChange={(e) => {
                            const raw = e.target.value;
                            setNested(part, "learning_rate", raw === "" ? 0.00005 : Number(raw));
                          }}
                        />
                        {lrHints.hint && (
                          <FieldHint hint={lrHints.hint} hintAnchor={lrHints.hintAnchor} />
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                </tr>
              );
              })}
            </tbody>
          </table>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t border-border">
          <NumberInput
            label="CLIP Skip"
            value={config.clip_skip ?? 2}
            onChange={(v) => set("clip_skip", v ?? 2)}
            min={1}
            step={1}
            placeholder="2"
            paramKey="clip_skip"
          />
        </div>
        <p className="text-xs text-muted">
          CLIP hidden layer used for text encoding during training and sampling. Default 2 matches Kohya.
        </p>
      </section>

      {/* Training Hyperparameters */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Training</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <NumberInput label="Epochs" value={config.epochs} onChange={(v) => set("epochs", v)} min={1} placeholder="30" paramKey="epochs" />
          <NumberInput label="Batch Size" value={config.batch_size} onChange={(v) => set("batch_size", v)} min={1} placeholder="1" paramKey="batch_size" />
          <NumberInput
            label="Grad Accumulation Steps"
            value={config.gradient_accumulation_steps}
            onChange={(v) => set("gradient_accumulation_steps", v)}
            min={1}
            placeholder="1"
            paramKey="gradient_accumulation_steps"
          />
          <SelectInput
            label="LR Scheduler"
            value={config.lr_scheduler ?? "constant"}
            onChange={(v) => set("lr_scheduler", v)}
            options={lrSchedulerOptions}
            paramKey="lr_scheduler"
          />
          <NumberInput
            label="LR Warmup Steps"
            value={config.lr_warmup_steps}
            onChange={(v) => set("lr_warmup_steps", v)}
            min={0}
            placeholder="0"
            paramKey="lr_warmup_steps"
          />
          <NumberInput
            label="Min SNR Gamma"
            value={config.min_snr_gamma}
            onChange={(v) => set("min_snr_gamma", v)}
            min={0}
            step={0.5}
            placeholder="5"
            paramKey="min_snr_gamma"
          />
          <NumberInput
            label="Noise Offset"
            value={config.noise_offset}
            onChange={(v) => set("noise_offset", v)}
            min={0}
            step={0.001}
            placeholder="0.0357"
            paramKey="noise_offset"
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
            paramKey="optimizer.type"
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
                paramKey="optimizer.weight_decay"
              />
              <NumberInput
                label="Beta 1"
                value={config.optimizer?.beta1}
                onChange={(v) => setNested("optimizer", "beta1", v)}
                min={0}
                max={0.999}
                step={0.01}
                placeholder="0.9"
                paramKey="optimizer.beta1"
              />
              <NumberInput
                label="Beta 2"
                value={config.optimizer?.beta2}
                onChange={(v) => setNested("optimizer", "beta2", v)}
                min={0}
                max={0.999}
                step={0.001}
                placeholder="0.999"
                paramKey="optimizer.beta2"
              />
            </>
          )}
          {isAdafactor && (
            <>
              <CheckboxInput
                label="Relative Step"
                checked={config.optimizer?.relative_step ?? false}
                onChange={(v) => setNested("optimizer", "relative_step", v)}
                paramKey="optimizer.relative_step"
              />
              <CheckboxInput
                label="Scale Parameter"
                checked={config.optimizer?.scale_parameter ?? false}
                onChange={(v) => setNested("optimizer", "scale_parameter", v)}
                paramKey="optimizer.scale_parameter"
              />
              <CheckboxInput
                label="Warmup Init"
                checked={config.optimizer?.warmup_init ?? false}
                onChange={(v) => setNested("optimizer", "warmup_init", v)}
                paramKey="optimizer.warmup_init"
              />
            </>
          )}
          {isProdigy && (
            <>
              <CheckboxInput
                label="Decouple"
                checked={config.optimizer?.decouple ?? true}
                onChange={(v) => setNested("optimizer", "decouple", v)}
                paramKey="optimizer.decouple"
              />
              <CheckboxInput
                label="Use Bias Correction"
                checked={config.optimizer?.use_bias_correction ?? true}
                onChange={(v) => setNested("optimizer", "use_bias_correction", v)}
                paramKey="optimizer.use_bias_correction"
              />
              <CheckboxInput
                label="Safeguard Warmup"
                checked={config.optimizer?.safeguard_warmup ?? true}
                onChange={(v) => setNested("optimizer", "safeguard_warmup", v)}
                paramKey="optimizer.safeguard_warmup"
              />
              <NumberInput
                label="d0"
                value={config.optimizer?.d0}
                onChange={(v) => setNested("optimizer", "d0", v)}
                min={0}
                step={0.00001}
                placeholder="0.00001"
                paramKey="optimizer.d0"
              />
              <NumberInput
                label="d Coef"
                value={config.optimizer?.d_coef}
                onChange={(v) => setNested("optimizer", "d_coef", v)}
                min={0}
                step={0.1}
                placeholder="1.0"
                paramKey="optimizer.d_coef"
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
          paramKey="resolution"
        />
        <label className="flex items-center gap-2 text-sm text-text mt-2 cursor-pointer">
          <input
            type="checkbox"
            checked={trainEnableBucket}
            onChange={(e) => set("enable_bucket", e.target.checked)}
            className="rounded"
          />
          <span className="flex items-center">
            Enable aspect-ratio bucketing
            {trainHint("enable_bucket").hint && (
              <FieldHint hint={trainHint("enable_bucket").hint!} hintAnchor="enable_bucket" />
            )}
          </span>
        </label>
        <div className="space-y-3 mt-2">
          <div className="text-xs font-medium text-muted">Concepts</div>
          {datasetsLoading ? (
            <div className="text-sm text-muted">Loading datasets…</div>
          ) : !datasets?.length ? (
            <div className="rounded-lg border border-dashed border-border p-6 text-center space-y-3">
              <p className="text-sm text-muted">
                No datasets yet. Create a dataset to specify training data.
              </p>
              <Link
                href="/datasets"
                className="inline-flex items-center gap-1.5 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium"
              >
                Create Dataset
              </Link>
            </div>
          ) : (
            <>
              {concepts.map((concept, i) => {
                const selectedDataset = datasetById(concept.dataset_id);
                return (
                  <div key={i} className="relative rounded-lg border border-border p-4 bg-bg">
                    <button
                      type="button"
                      onClick={() => removeConcept(i)}
                      className="absolute top-2 right-2 p-1 rounded hover:bg-white/10 text-muted hover:text-error"
                    >
                      <X size={13} />
                    </button>
                    <div className="text-xs text-muted mb-3">Concept {i + 1}</div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="md:col-span-1">
                        <SelectInput
                          label="Dataset"
                          value={concept.dataset_id != null ? String(concept.dataset_id) : ""}
                          onChange={(v) => updateConcept(i, "dataset_id", Number(v))}
                          options={[{ value: "", label: "Select dataset…" }, ...datasetOptions]}
                          paramKey="concepts.dataset_id"
                        />
                        {selectedDataset && (
                          <p className="text-xs text-muted mt-1 break-all">{selectedDataset.image_dir}</p>
                        )}
                        {selectedDataset && !isDatasetCompatible(selectedDataset) && (
                          <p className="text-xs text-warning mt-1">
                            Dataset must be prepared at {trainResolution}px. Open the dataset page to crop and bake
                            images.
                          </p>
                        )}
                        {concept.dataset_id == null && (
                          <p className="text-xs text-error mt-1">Select a dataset</p>
                        )}
                        {concept.dataset_id != null && !selectedDataset && (
                          <p className="text-xs text-error mt-1">Dataset not found</p>
                        )}
                      </div>
                      <Field label="Trigger Words" {...trainHint("concepts.trigger_words")}>
                        <input
                          type="text"
                          className={inputClassName}
                          value={(concept.trigger_words ?? []).join(", ")}
                          onChange={(e) =>
                            updateConcept(
                              i,
                              "trigger_words",
                              e.target.value
                                .split(",")
                                .map((word) => word.trim())
                                .filter(Boolean)
                            )
                          }
                          placeholder="ohwx, person"
                        />
                      </Field>
                      <Field label="Caption Extension" {...trainHint("concepts.caption_extension")}>
                        <input
                          type="text"
                          className={inputClassName}
                          value={concept.caption_extension ?? ".txt"}
                          onChange={(e) => updateConcept(i, "caption_extension", e.target.value)}
                          placeholder=".txt"
                        />
                      </Field>
                      <Field label="Repeats" {...trainHint("concepts.repeats")}>
                        <input
                          type="number"
                          className={inputClassName}
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
                className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border hover:border-text/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
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
            paramKey="mixed_precision"
          />
          <NumberInput
            label="Seed (optional)"
            value={config.seed ?? null}
            onChange={(v) => set("seed", v)}
            min={0}
            placeholder="random"
            paramKey="seed"
          />
          <div className="flex items-center pb-1">
            <CheckboxInput
              label="Gradient Checkpointing"
              checked={config.gradient_checkpointing ?? true}
              onChange={(v) => set("gradient_checkpointing", v)}
              paramKey="gradient_checkpointing"
            />
          </div>
        </div>
      </section>

      {/* Performance */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Performance</div>

        {/* Caching */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted">Caching</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
            <CheckboxInput
              label="Cache Latents (RAM)"
              checked={cacheLatentsEnabled}
              onChange={(v) => set("cache_latents", v)}
              paramKey="cache_latents"
            />
            <CheckboxInput
              label="Cache Text Encoder Outputs (RAM)"
              checked={cacheTextEncoderEnabled}
              onChange={(v) => set("cache_text_encoder_outputs", v)}
              disabled={textEncoderTrainingEnabled}
              paramKey="cache_text_encoder_outputs"
            />
            <div className="space-y-1">
              <CheckboxInput
                label="Cache Latents to Disk (.npz)"
                checked={config.cache_latents_to_disk ?? false}
                onChange={(v) => set("cache_latents_to_disk", v)}
                disabled={!cacheLatentsEnabled}
                paramKey="cache_latents_to_disk"
              />
              {!cacheLatentsEnabled && (
                <p className="text-xs text-muted">Requires RAM caching to be enabled.</p>
              )}
            </div>
            <div className="space-y-1">
              <CheckboxInput
                label="Cache Text Encoder Outputs to Disk"
                checked={config.cache_text_encoder_outputs_to_disk ?? false}
                onChange={(v) => set("cache_text_encoder_outputs_to_disk", v)}
                disabled={!cacheTextEncoderEnabled}
                paramKey="cache_text_encoder_outputs_to_disk"
              />
              {!cacheTextEncoderEnabled && !textEncoderTrainingEnabled && (
                <p className="text-xs text-muted">Requires RAM caching to be enabled.</p>
              )}
            </div>
          </div>
          {textEncoderTrainingEnabled && (
            <p className="text-xs text-muted mt-1">
              Text encoder output caching is disabled while training text encoders.
            </p>
          )}
        </div>

        {/* Attention & Precision */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end mt-4">
          <SelectInput
            label="Attention Mechanism"
            value={config.attention_mechanism ?? "sdpa"}
            onChange={(v) => set("attention_mechanism", v)}
            paramKey="attention_mechanism"
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
              paramKey="tf32"
            />
            <CheckboxInput
              label="torch.compile (slower start)"
              checked={config.torch_compile ?? false}
              onChange={(v) => set("torch_compile", v)}
              paramKey="torch_compile"
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
            paramKey="num_dataloader_workers"
          />
          <div className="flex items-end pb-1">
            <CheckboxInput
              label="Pin Memory (requires workers > 0)"
              checked={config.dataloader_pin_memory ?? true}
              onChange={(v) => set("dataloader_pin_memory", v)}
              paramKey="dataloader_pin_memory"
            />
          </div>
        </div>
      </section>

      {/* Checkpointing */}
      <section className={sectionClass}>
        <div className={sectionTitleClass}>Checkpointing</div>
        <div className="space-y-4">
          <CheckboxInput
            label="Enable intermediate checkpoints"
            checked={checkpointingEnabled}
            onChange={(v) => set("checkpointing_enabled", v)}
            paramKey="checkpointing_enabled"
          />
          <NumberInput
            label="Save Every N Epochs"
            value={config.save_every_n_epochs}
            onChange={(v) => set("save_every_n_epochs", v)}
            min={1}
            placeholder="1"
            disabled={!checkpointingEnabled}
            paramKey="save_every_n_epochs"
          />
        </div>
      </section>

      {/* Sampling */}
      <section className={`${sectionClass} ${!checkpointingEnabled ? "opacity-60" : ""}`}>
        <div className={sectionTitleClass}>Sampling</div>
        <div className="space-y-4">
          <CheckboxInput
            label="Run sampling after training for intermediate checkpoints"
            checked={samplingEnabled}
            onChange={(v) => set("sampling_enabled", v)}
            disabled={!checkpointingEnabled}
            paramKey="sampling_enabled"
          />
          {!checkpointingEnabled && (
            <p className="text-xs text-muted">Sampling requires checkpointing to be enabled.</p>
          )}
          {checkpointingEnabled && samplingEnabled && (
            <div className="space-y-3">
              <div className="text-xs font-medium text-muted">Sampling Config</div>
              {samplingConfigsLoading ? (
                <div className="text-sm text-muted">Loading sampling configs…</div>
              ) : !samplingConfigs?.length ? (
                <div className="rounded-lg border border-dashed border-border p-6 text-center space-y-3">
                  <p className="text-sm text-muted">
                    No sampling configs yet. Create one to configure preview prompts and sampler settings.
                  </p>
                  <Link
                    href="/configs/new?type=sampling"
                    className="inline-flex items-center gap-1.5 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium"
                  >
                    Create Sampling Config
                  </Link>
                </div>
              ) : (
                <>
                  <SelectInput
                    label="Sampling Config"
                    value={config.sampling_config_id != null ? String(config.sampling_config_id) : ""}
                    onChange={(v) => set("sampling_config_id", v ? Number(v) : null)}
                    options={[{ value: "", label: "None" }, ...samplingConfigOptions]}
                    paramKey="sampling_config_id"
                  />
                  {samplingEnabled && selectedSamplingPreview && (
                    <p className="text-xs text-muted">
                      {selectedSamplingPreview.promptCount} prompt(s), {selectedSamplingPreview.steps} steps,{" "}
                      {selectedSamplingPreview.scheduler} scheduler
                    </p>
                  )}
                  {samplingEnabled && config.sampling_config_id != null && !selectedSamplingConfig && (
                    <p className="text-xs text-error">Sampling config not found</p>
                  )}
                  {samplingConfigRequired && config.sampling_config_id == null && (
                    <p className="text-xs text-error">Select a sampling config when sampling is enabled</p>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
