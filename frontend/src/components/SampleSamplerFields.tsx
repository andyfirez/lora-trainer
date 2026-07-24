"use client";

import { Field } from "@/components/ui/Field";
import { Select } from "@/components/ui/Select";
import { comfySamplerNameOptions, comfySchedulerOptions } from "@/lib/sampleSamplerOptions";

interface SampleSamplerFieldsProps {
  config: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

export default function SampleSamplerFields({ config, onChange }: SampleSamplerFieldsProps) {
  return (
    <>
      <Field label="Sample Sampler">
        <Select
          value={(config.sample_sampler_name as string) ?? "euler"}
          onChange={(v) => onChange("sample_sampler_name", v)}
          options={comfySamplerNameOptions}
        />
      </Field>
      <Field label="Sample Sigma Scheduler">
        <Select
          value={(config.sample_scheduler as string) ?? "simple"}
          onChange={(v) => onChange("sample_scheduler", v)}
          options={comfySchedulerOptions}
        />
      </Field>
    </>
  );
}
