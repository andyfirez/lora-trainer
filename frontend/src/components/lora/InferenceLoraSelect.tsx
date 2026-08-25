"use client";

import { useMemo } from "react";
import { inputClassName } from "@/components/ui/Input";
import { useInferenceLoras } from "@/hooks/useInferenceLoras";

interface InferenceLoraSelectProps {
  value: string;
  onChange: (relativePath: string) => void;
  disabled?: boolean;
  className?: string;
  id?: string;
}

export default function InferenceLoraSelect({
  value,
  onChange,
  disabled = false,
  className,
  id,
}: InferenceLoraSelectProps) {
  const { data, isLoading, error } = useInferenceLoras();

  const options = useMemo(() => data?.loras ?? [], [data?.loras]);
  const knownPath = value.trim();
  const inCatalog = !knownPath || options.some((item) => item.relative_path === knownPath);

  return (
    <div className={className}>
      <select
        id={id}
        className={inputClassName}
        value={knownPath}
        disabled={disabled || isLoading}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Base model only</option>
        {options.map((item) => (
          <option key={item.relative_path} value={item.relative_path}>
            {item.name}
          </option>
        ))}
      </select>
      {error ? <p className="mt-1 text-xs text-error">Failed to load LoRA catalog</p> : null}
      {!inCatalog && knownPath ? (
        <p className="mt-1 text-xs text-warning">Path not in inference catalog: {knownPath}</p>
      ) : null}
    </div>
  );
}

export function inferenceLoraNameForPath(
  relativePath: string,
  loras: { name: string; relative_path: string }[],
): string | null {
  const match = loras.find((item) => item.relative_path === relativePath);
  return match?.name ?? null;
}
