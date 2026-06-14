"use client";

import {
  diffusersSchedulerOptions,
  reforgeSamplerOptions,
  schedulerModeOptions,
} from "@/lib/sampleSamplerOptions";

const selectClass =
  "w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-1.5 text-sm text-white focus:outline-none focus:border-[var(--accent)]";
const labelClass = "block text-xs font-medium text-[var(--muted)] mb-1";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      {children}
    </div>
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

export interface SampleSamplerFieldsProps {
  useReforgeSampler: boolean;
  sampleScheduler: string;
  sampleSampler: string;
  sampleSchedulerMode: string;
  onChange: (key: string, value: unknown) => void;
  reforgeCheckboxLabel: string;
}

export default function SampleSamplerFields({
  useReforgeSampler,
  sampleScheduler,
  sampleSampler,
  sampleSchedulerMode,
  onChange,
  reforgeCheckboxLabel,
}: SampleSamplerFieldsProps) {
  return (
    <div className="col-span-full space-y-3">
      <CheckboxInput
        label={reforgeCheckboxLabel}
        checked={useReforgeSampler}
        onChange={(v) => onChange("use_reforge_sampler", v)}
      />
      {useReforgeSampler ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SelectInput
            label="Sampler"
            value={sampleSampler}
            onChange={(v) => onChange("sample_sampler", v)}
            options={reforgeSamplerOptions}
          />
          <SelectInput
            label="Scheduler Mode"
            value={sampleSchedulerMode}
            onChange={(v) => onChange("sample_scheduler_mode", v)}
            options={schedulerModeOptions}
          />
        </div>
      ) : (
        <SelectInput
          label="Sample Scheduler"
          value={sampleScheduler}
          onChange={(v) => onChange("sample_scheduler", v)}
          options={diffusersSchedulerOptions}
        />
      )}
    </div>
  );
}
