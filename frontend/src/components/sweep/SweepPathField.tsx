"use client";

import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import ModeToggle from "@/components/sweep/ModeToggle";
import { labelClassName } from "@/components/ui/Input";
import type { SweepMode, SweepParameter } from "@/lib/sweepUtils";

interface SweepPathFieldProps {
  label: string;
  param: SweepParameter;
  onChange: (param: SweepParameter) => void;
  placeholder?: string;
  pickerTitle?: string;
}

export default function SweepPathField({
  label,
  param,
  onChange,
  placeholder = "Path to .safetensors",
  pickerTitle = "Select LoRA",
}: SweepPathFieldProps) {
  const mode = param.mode ?? "fixed";

  function setMode(next: SweepMode) {
    if (next === "vary") {
      const existing = param.values?.length
        ? param.values.map(String)
        : param.value != null && String(param.value).trim()
          ? [String(param.value)]
          : [""];
      onChange({ mode: "vary", values: existing });
    } else {
      const first = param.values?.find((v) => String(v).trim()) ?? param.value ?? "";
      onChange({ mode: "fixed", value: first ? String(first) : null });
    }
  }

  function updateFixed(value: string) {
    onChange({ mode: "fixed", value: value || null });
  }

  function updateValue(i: number, value: string) {
    const values = [...(param.values ?? [])];
    values[i] = value;
    onChange({ mode: "vary", values });
  }

  function addValue() {
    onChange({ mode: "vary", values: [...(param.values ?? []), ""] });
  }

  function removeValue(i: number) {
    const values = (param.values ?? []).filter((_, idx) => idx !== i);
    onChange({
      mode: "vary",
      values: values.length ? values : [""],
    });
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label className={labelClassName}>{label}</label>
        <ModeToggle mode={mode} onChange={setMode} />
      </div>
      {mode === "fixed" ? (
        <PathInput
          label=""
          value={String(param.value ?? "")}
          onChange={updateFixed}
          placeholder={placeholder}
          pickerTitle={pickerTitle}
          kind="file"
        />
      ) : (
        <div className="rounded-lg border border-border/60 bg-bg/40 p-3 space-y-2">
          {(param.values ?? []).map((value, i) => (
            <div key={i} className="flex items-center gap-2 min-w-0">
              <div className="flex-1 min-w-0">
                <PathInput
                  label=""
                  value={String(value ?? "")}
                  onChange={(v) => updateValue(i, v)}
                  placeholder={placeholder}
                  pickerTitle={pickerTitle}
                  kind="file"
                />
              </div>
              <button
                type="button"
                onClick={() => removeValue(i)}
                className="p-1.5 rounded hover:bg-white/10 text-muted hover:text-error shrink-0"
              >
                <X size={13} />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addValue}
            className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border hover:border-text/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
          >
            <Plus size={13} /> Add value
          </button>
        </div>
      )}
    </div>
  );
}
