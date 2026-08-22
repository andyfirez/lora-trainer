"use client";

import { useEffect, useMemo, useState } from "react";
import { Plus, X } from "lucide-react";
import PathInput from "@/components/PathInput";
import ModeToggle from "@/components/sweep/ModeToggle";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import type { LoraEntry, SweepMode, SweepParameter } from "@/lib/sweepUtils";
import {
  emptyLoraEntry,
  normalizeLoraEntry,
  parseLoraEntries,
  parseLoraEntry,
} from "@/lib/sweepUtils";

interface LoraEntryFieldProps {
  label: string;
  param: SweepParameter;
  onChange: (param: SweepParameter) => void;
  allowVary?: boolean;
  variant?: "sweep" | "stack";
}

export default function LoraEntryField({
  label,
  param,
  onChange,
  allowVary = true,
  variant = "sweep",
}: LoraEntryFieldProps) {
  if (variant === "stack") {
    return <LoraStackField label={label} param={param} onChange={onChange} />;
  }

  const mode = allowVary ? (param.mode ?? "fixed") : "fixed";

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
        {allowVary ? <ModeToggle mode={mode} onChange={setMode} /> : null}
      </div>
      {mode === "fixed" ? (
        <LoraEntryRow entry={fixedEntry} onChange={updateFixed} showTrigger={allowVary} />
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

function loraEntryPath(entry: LoraEntry): string | null {
  const path = entry.path == null ? null : String(entry.path).trim();
  return path || null;
}

function LoraStackField({
  label,
  param,
  onChange,
}: {
  label: string;
  param: SweepParameter;
  onChange: (param: SweepParameter) => void;
}) {
  const rowsFromParam = useMemo(
    () =>
      parseLoraEntries(param.value)
        .map(normalizeLoraEntry)
        .filter((entry) => loraEntryPath(entry)),
    [param.value],
  );
  const [draftRows, setDraftRows] = useState<LoraEntry[]>([]);

  useEffect(() => {
    setDraftRows((drafts) => drafts.filter((entry) => !loraEntryPath(entry)));
  }, [param.value]);

  const rows = [...rowsFromParam, ...draftRows];

  function persistAll(allRows: LoraEntry[]) {
    const withPath = allRows.filter((entry) => loraEntryPath(entry));
    const withoutPath = allRows.filter((entry) => !loraEntryPath(entry));
    setDraftRows(withoutPath);
    if (withPath.length === 0) {
      onChange({ mode: "fixed", value: emptyLoraEntry() });
      return;
    }
    if (withPath.length === 1 && withoutPath.length === 0) {
      onChange({ mode: "fixed", value: withPath[0] });
      return;
    }
    onChange({ mode: "fixed", value: [...withPath, ...withoutPath] });
  }

  function updateRow(index: number, entry: LoraEntry) {
    persistAll(rows.map((row, i) => (i === index ? entry : row)));
  }

  function addRow() {
    setDraftRows((prev) => [...prev, emptyLoraEntry()]);
  }

  function removeRow(index: number) {
    persistAll(rows.filter((_, i) => i !== index));
  }

  return (
    <div>
      <label className={labelClassName}>{label}</label>
      <div className="space-y-2">
        {rows.map((entry, i) => (
          <div key={i} className="flex items-start gap-2 min-w-0">
            <div className="flex-1 min-w-0">
              <LoraEntryRow
                entry={normalizeLoraEntry(entry)}
                onChange={(next) => updateRow(i, next)}
                showTrigger
                showWeight
              />
            </div>
            <button
              type="button"
              onClick={() => removeRow(i)}
              className="p-1.5 rounded hover:bg-white/10 text-muted hover:text-error shrink-0 mt-1"
            >
              <X size={13} />
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={addRow}
          className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border hover:border-text/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
        >
          <Plus size={13} /> Add LoRA
        </button>
      </div>
    </div>
  );
}

function LoraEntryRow({
  entry,
  onChange,
  showTrigger = false,
  showWeight = false,
}: {
  entry: LoraEntry;
  onChange: (entry: LoraEntry) => void;
  showTrigger?: boolean;
  showWeight?: boolean;
}) {
  const hasPath = Boolean(entry.path?.trim());
  const triggerVisible = showTrigger && hasPath;

  function updatePath(path: string) {
    const normalized = path.trim() ? path : null;
    onChange({
      ...entry,
      path: normalized,
      trigger: normalized ? entry.trigger ?? "" : "",
      weight: entry.weight ?? 1,
    });
  }

  return (
    <div className="min-w-0 space-y-2">
      <div className="flex min-w-0 items-center gap-2">
        <div className="min-w-0 flex-1">
          <PathInput
            label=""
            value={entry.path ?? ""}
            onChange={updatePath}
            placeholder="Leave empty for base model only"
            pickerTitle="Select LoRA"
            kind="file"
          />
        </div>
        {showWeight ? (
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            className="w-16 shrink-0 rounded-lg border border-border bg-input px-2 py-2 text-sm text-text focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
            value={entry.weight ?? 1}
            onChange={(e) =>
              onChange({
                ...entry,
                weight: e.target.value === "" ? 1 : Number(e.target.value),
              })
            }
            title="LoRA weight"
          />
        ) : null}
      </div>
      {triggerVisible ? (
        <input
          type="text"
          className={inputClassName}
          value={entry.trigger ?? ""}
          onChange={(e) => onChange({ ...entry, trigger: e.target.value })}
          placeholder="Trigger words (comma-separated)"
        />
      ) : null}
    </div>
  );
}
