"use client";

import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import ModeToggle from "@/components/sweep/ModeToggle";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import type { LoraEntry, SweepMode, SweepParameter } from "@/lib/sweepUtils";
import { emptyLoraEntry, normalizeLoraEntry, parseLoraEntry } from "@/lib/sweepUtils";

interface LoraEntryFieldProps {
  label: string;
  param: SweepParameter;
  onChange: (param: SweepParameter) => void;
}

export default function LoraEntryField({ label, param, onChange }: LoraEntryFieldProps) {
  const mode = param.mode ?? "fixed";

  function setMode(next: SweepMode) {
    if (next === "vary") {
      const existing = param.values?.length
        ? param.values.map((v) => normalizeLoraEntry(parseLoraEntry(v)))
        : [normalizeLoraEntry(parseLoraEntry(param.value))];
      onChange({ mode: "vary", values: existing });
    } else {
      const first =
        param.values?.map((v) => parseLoraEntry(v)).find((e) => e.path) ??
        parseLoraEntry(param.value);
      onChange({ mode: "fixed", value: normalizeLoraEntry(first) });
    }
  }

  function updateFixed(entry: LoraEntry) {
    onChange({ mode: "fixed", value: normalizeLoraEntry(entry) });
  }

  function updateValue(i: number, entry: LoraEntry) {
    const values = [...(param.values ?? [])];
    values[i] = normalizeLoraEntry(entry);
    onChange({ mode: "vary", values });
  }

  function addValue() {
    onChange({ mode: "vary", values: [...(param.values ?? []), emptyLoraEntry()] });
  }

  function removeValue(i: number) {
    const values = (param.values ?? []).filter((_, idx) => idx !== i);
    onChange({ mode: "vary", values: values.length ? values : [emptyLoraEntry()] });
  }

  const fixedEntry = normalizeLoraEntry(parseLoraEntry(param.value));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label className={labelClassName}>{label}</label>
        <ModeToggle mode={mode} onChange={setMode} />
      </div>
      {mode === "fixed" ? (
        <LoraEntryRow entry={fixedEntry} onChange={updateFixed} showTrigger={false} />
      ) : (
        <div className="rounded-lg border border-border/60 bg-bg/40 p-3 space-y-2">
          {(param.values ?? []).map((value, i) => (
            <div key={i} className="flex items-start gap-2 min-w-0">
              <div className="flex-1 min-w-0">
                <LoraEntryRow
                  entry={normalizeLoraEntry(parseLoraEntry(value))}
                  onChange={(entry) => updateValue(i, entry)}
                  showTrigger
                />
              </div>
              <button
                type="button"
                onClick={() => removeValue(i)}
                className="p-1.5 rounded hover:bg-white/10 text-muted hover:text-error shrink-0 mt-1"
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
            <Plus size={13} /> Add LoRA
          </button>
        </div>
      )}
    </div>
  );
}

function LoraEntryRow({
  entry,
  onChange,
  showTrigger = false,
}: {
  entry: LoraEntry;
  onChange: (entry: LoraEntry) => void;
  showTrigger?: boolean;
}) {
  const hasPath = Boolean(entry.path?.trim());
  const triggerVisible = showTrigger && hasPath;

  function updatePath(path: string) {
    const normalized = path.trim() ? path : null;
    onChange({
      ...entry,
      path: normalized,
      trigger: normalized ? entry.trigger ?? "" : "",
    });
  }

  return (
    <div className={`min-w-0 ${triggerVisible ? "flex items-start gap-2" : ""}`}>
      <div className={triggerVisible ? "flex-1 min-w-0" : "w-full min-w-0"}>
        <PathInput
          label=""
          value={entry.path ?? ""}
          onChange={updatePath}
          placeholder="Leave empty for base model only"
          pickerTitle="Select LoRA"
          kind="file"
        />
      </div>
      {triggerVisible ? (
        <input
          type="text"
          className={`${inputClassName} flex-1 min-w-0 mt-0`}
          value={entry.trigger ?? ""}
          onChange={(e) => onChange({ ...entry, trigger: e.target.value })}
          placeholder="Trigger words (comma-separated)"
        />
      ) : null}
    </div>
  );
}
