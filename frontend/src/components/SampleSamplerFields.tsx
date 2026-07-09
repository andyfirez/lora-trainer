"use client";

import { labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { diffusersSchedulerOptions } from "@/lib/sampleSamplerOptions";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className={labelClassName}>{label}</label>
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
