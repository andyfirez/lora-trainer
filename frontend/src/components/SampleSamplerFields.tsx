"use client";

import { diffusersSchedulerOptions } from "@/lib/sampleSamplerOptions";

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

export interface SampleSamplerFieldsProps {
  sampleScheduler: string;
  onChange: (key: string, value: unknown) => void;
}

export default function SampleSamplerFields({ sampleScheduler, onChange }: SampleSamplerFieldsProps) {
  return (
    <div className="col-span-full">
      <SelectInput
        label="Sample Scheduler"
        value={sampleScheduler}
        onChange={(v) => onChange("sample_scheduler", v)}
        options={diffusersSchedulerOptions}
      />
    </div>
  );
}
