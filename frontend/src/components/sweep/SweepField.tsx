"use client";

import { Plus, X } from "lucide-react";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import type { SelectOption } from "@/lib/sampleSamplerOptions";
import type { SweepMode, SweepParameter } from "@/lib/sweepUtils";

interface SweepFieldProps {
  label: string;
  param: SweepParameter;
  onChange: (param: SweepParameter) => void;
  type?: "text" | "number" | "select";
  placeholder?: string;
  multiline?: boolean;
  selectOptions?: SelectOption[];
}

export default function SweepField({
  label,
  param,
  onChange,
  type = "text",
  placeholder,
  multiline = false,
  selectOptions = [],
}: SweepFieldProps) {
  const mode = param.mode ?? "fixed";
  const defaultSelectValue = selectOptions[0]?.value ?? "";

  function renderSelect(value: unknown, onSelect: (v: string) => void) {
    const selected = String(value ?? defaultSelectValue);
    return (
      <select
        className={`${selectClassName} w-full`}
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
      >
        {selectOptions.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    );
  }

  function setMode(next: SweepMode) {
    if (next === "vary") {
      onChange({
        mode: "vary",
        values: param.values?.length
          ? param.values
          : [param.value ?? (type === "select" ? defaultSelectValue : type === "number" ? 0 : "")],
      });
    } else {
      const first =
        param.values?.[0] ??
        param.value ??
        (type === "select" ? defaultSelectValue : type === "number" ? 0 : "");
      onChange({ mode: "fixed", value: first });
    }
  }

  function updateFixed(value: unknown) {
    onChange({ mode: "fixed", value: type === "number" ? Number(value) : value });
  }

  function updateValue(i: number, value: unknown) {
    const values = [...(param.values ?? [])];
    values[i] = type === "number" ? Number(value) : value;
    onChange({ mode: "vary", values });
  }

  function addValue() {
    onChange({
      mode: "vary",
      values: [
        ...(param.values ?? []),
        type === "select" ? defaultSelectValue : type === "number" ? 0 : "",
      ],
    });
  }

  function removeValue(i: number) {
    const values = (param.values ?? []).filter((_, idx) => idx !== i);
    onChange({
      mode: "vary",
      values: values.length
        ? values
        : [type === "select" ? defaultSelectValue : type === "number" ? 0 : ""],
    });
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label className={labelClassName}>{label}</label>
        <div className="flex rounded-lg border border-border overflow-hidden text-xs">
          <button
            type="button"
            onClick={() => setMode("fixed")}
            className={`px-3 py-1 ${mode === "fixed" ? "bg-accent text-white" : "text-muted hover:text-text"}`}
          >
            Fixed
          </button>
          <button
            type="button"
            onClick={() => setMode("vary")}
            className={`px-3 py-1 ${mode === "vary" ? "bg-accent text-white" : "text-muted hover:text-text"}`}
          >
            Vary
          </button>
        </div>
      </div>
      {mode === "fixed" ? (
        type === "select" ? (
          renderSelect(param.value, (v) => updateFixed(v))
        ) : multiline ? (
          <textarea
            className={`${inputClassName} min-h-[72px]`}
            value={String(param.value ?? "")}
            placeholder={placeholder}
            onChange={(e) => updateFixed(e.target.value)}
          />
        ) : (
          <input
            type={type}
            className={inputClassName}
            value={
              param.value === null || param.value === undefined
                ? ""
                : (param.value as string | number)
            }
            placeholder={placeholder}
            step={type === "number" ? "any" : undefined}
            onChange={(e) =>
              updateFixed(e.target.value === "" && type === "number" ? null : e.target.value)
            }
          />
        )
      ) : (
        <div className="space-y-2">
          {(param.values ?? []).map((value, i) => (
            <div key={i} className="flex items-center gap-2 min-w-0">
              {type === "select" ? (
                <div className="flex-1 min-w-0">{renderSelect(value, (v) => updateValue(i, v))}</div>
              ) : multiline ? (
                <textarea
                  className={`${inputClassName} min-h-[56px]`}
                  value={String(value ?? "")}
                  placeholder={placeholder}
                  onChange={(e) => updateValue(i, e.target.value)}
                />
              ) : (
                <input
                  type={type}
                  className={inputClassName}
                  value={value as string | number}
                  placeholder={placeholder}
                  step={type === "number" ? "any" : undefined}
                  onChange={(e) =>
                    updateValue(i, e.target.value === "" && type === "number" ? null : e.target.value)
                  }
                />
              )}
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
