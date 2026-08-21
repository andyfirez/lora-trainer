"use client";

import FieldHint from "@/components/FieldHint";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { trainHint } from "@/lib/trainParameterMetadata";

export function TrainField({
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

export function TrainTextInput({
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
    <TrainField label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
      <input
        type="text"
        className={inputClassName}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </TrainField>
  );
}

export function TrainNumberInput({
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
    <TrainField label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
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
    </TrainField>
  );
}

export function TrainSelectInput({
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
    <TrainField label={label} hint={hints.hint} hintAnchor={hints.hintAnchor}>
      <select className={selectClassName} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </TrainField>
  );
}

export function TrainCheckboxInput({
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

export const weightDtypeOptions = [
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

export const lrSchedulerOptions = [
  { value: "constant", label: "Constant" },
  { value: "constant_with_warmup", label: "Constant with Warmup" },
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
  { value: "cosine_with_restarts", label: "Cosine with Restarts" },
  { value: "polynomial", label: "Polynomial" },
];
