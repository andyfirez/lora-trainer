"use client";

import FieldHint from "@/components/FieldHint";
import { labelClassName } from "@/components/ui/Input";
import { selectClassName } from "@/components/ui/Select";
import { trainHint } from "@/lib/trainParameterMetadata";
import { cn } from "@/lib/cn";

const inheritedHintSuffix =
  " When not overridden in this config, uses the current value from Settings → GPU.";

function inheritedHint(paramKey?: string): { hint?: string; hintAnchor?: string } {
  const hints = paramKey ? trainHint(paramKey) : {};
  if (!hints.hint) {
    return {
      hint: `Uses the current value from Settings → GPU when not overridden in this config.`,
      hintAnchor: hints.hintAnchor,
    };
  }
  return {
    hint: `${hints.hint}${inheritedHintSuffix}`,
    hintAnchor: hints.hintAnchor,
  };
}

interface InheritedSelectFieldProps {
  label: string;
  value: string | undefined | null;
  globalDefault: string;
  options: { value: string; label: string }[];
  onChange: (value: string | undefined) => void;
  paramKey?: string;
}

export function InheritedSelectField({
  label,
  value,
  globalDefault,
  options,
  onChange,
  paramKey,
}: InheritedSelectFieldProps) {
  const inherited = value == null || value === "";
  const displayValue = inherited ? globalDefault : value;
  const hints = inheritedHint(paramKey);

  return (
    <div>
      <div className="flex items-center mb-1">
        <label className={`${labelClassName} mb-0`}>{label}</label>
        {hints.hint && <FieldHint hint={hints.hint} hintAnchor={hints.hintAnchor} />}
      </div>
      <select
        className={cn(selectClassName, inherited && "text-muted")}
        value={displayValue}
        onChange={(e) => {
          const next = e.target.value;
          onChange(next === globalDefault ? undefined : next);
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface InheritedCheckboxFieldProps {
  label: string;
  value: boolean | undefined | null;
  globalDefault: boolean;
  onChange: (value: boolean | undefined) => void;
  paramKey?: string;
}

export function InheritedCheckboxField({
  label,
  value,
  globalDefault,
  onChange,
  paramKey,
}: InheritedCheckboxFieldProps) {
  const inherited = value == null;
  const displayValue = inherited ? globalDefault : value;
  const hints = inheritedHint(paramKey);

  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <input
        type="checkbox"
        className="w-4 h-4 rounded accent-accent"
        checked={displayValue}
        onChange={(e) => {
          const next = e.target.checked;
          onChange(next === globalDefault ? undefined : next);
        }}
      />
      <span className={cn("text-sm flex items-center", inherited ? "text-muted" : "text-text")}>
        {label}
        {hints.hint && <FieldHint hint={hints.hint} hintAnchor={hints.hintAnchor} />}
      </span>
    </label>
  );
}
