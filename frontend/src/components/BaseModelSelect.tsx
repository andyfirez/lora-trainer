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
  const { options, isLoading, error } = useBaseModelOptions(value ? [value] : []);

  return (
    <div className={className}>
      <div className="flex items-center">
        <label className={labelClassName}>{label}</label>
        {hint ? <FieldHint hint={hint} hintAnchor={hintAnchor} /> : null}
      </div>
      <select
        className={selectClassName}
        value={value ?? ""}
        disabled={isLoading}
        onChange={(event) => onChange(event.target.value)}
      >
        {isLoading ? (
          <option value={value ?? ""}>Loading models…</option>
        ) : (
          <>
            {!value ? <option value="">Select a base model…</option> : null}
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
    </div>
  );
}
