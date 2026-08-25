"use client";

import FieldHint from "@/components/FieldHint";
import { labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { useBaseModelOptions } from "@/hooks/useBaseModelOptions";

interface BaseModelSelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  hintAnchor?: string;
  className?: string;
}

export default function BaseModelSelect({
  label,
  value,
  onChange,
  hint,
  hintAnchor,
  className,
}: BaseModelSelectProps) {
  const { options, isLoading, error } = useBaseModelOptions();
  const trimmed = value.trim();
  const knownValue = trimmed && options.some((option) => option.value === trimmed);
  const selectValue = knownValue ? trimmed : "";

  return (
    <div className={className}>
      <div className="flex items-center">
        <label className={labelClassName}>{label}</label>
        {hint ? <FieldHint hint={hint} hintAnchor={hintAnchor} /> : null}
      </div>
      <select
        className={selectClassName}
        value={isLoading ? "" : selectValue}
        disabled={isLoading}
        onChange={(event) => onChange(event.target.value)}
      >
        {isLoading ? (
          <option value="">Loading models…</option>
        ) : (
          <>
            <option value="">Select a base model…</option>
            {options.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </>
        )}
      </select>
      {error ? (
        <p className="mt-1 text-xs text-error">
          {error instanceof Error ? error.message : "Failed to load base models"}
        </p>
      ) : null}
      {!isLoading && trimmed && !knownValue ? (
        <p className="mt-1 text-xs text-warning">Not in base models folder: {trimmed}</p>
      ) : null}
    </div>
  );
}
