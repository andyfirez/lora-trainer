import type { CSSProperties } from "react";
import { labelClassName } from "@/components/ui/Input";
import { cn } from "@/lib/cn";

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  className?: string;
}

export default function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  className,
}: SliderProps) {
  const decimals = String(step).includes(".") ? String(step).split(".")[1]?.length ?? 0 : 0;
  const span = max - min;
  const progress = span === 0 ? 0 : ((value - min) / span) * 100;

  function parse(raw: string): number {
    const next = Number(raw);
    if (!Number.isFinite(next)) return value;
    const snapped = Math.round(next / step) * step;
    return Math.min(max, Math.max(min, Number(snapped.toFixed(decimals))));
  }

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-center justify-between gap-2">
        <label className={`${labelClassName} mb-0`}>{label}</label>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          className="w-16 rounded-md border border-border bg-input px-1.5 py-0.5 text-right text-xs text-text focus:border-accent focus:outline-none"
          value={decimals ? value.toFixed(decimals) : String(value)}
          onChange={(event) => onChange(parse(event.target.value))}
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(parse(event.target.value))}
        className="slider-input"
        style={{ "--slider-progress": `${progress}%` } as CSSProperties}
      />
    </div>
  );
}
