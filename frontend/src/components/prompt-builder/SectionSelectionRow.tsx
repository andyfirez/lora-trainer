"use client";

import { ChevronDown, GripVertical, Trash2, X } from "lucide-react";
import { useState } from "react";
import { CREATE_OPTION } from "@/components/prompt-builder/constants";
import InlineRenameField from "@/components/prompt-builder/InlineRenameField";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import Modal, { ModalFooter } from "@/components/ui/Modal";
import { selectClassName } from "@/components/ui/Select";
import type { PromptSelectionMode, PromptTemplateSection } from "@/types/promptBuilder";
import { cn } from "@/lib/cn";

interface SectionSelectionRowProps {
  section: PromptTemplateSection;
  onChange: (body: {
    selection_mode?: PromptSelectionMode;
    selected_value_id?: number | null;
    selected_value_ids?: number[];
    clear_selected_value?: boolean;
    include_empty?: boolean;
  }) => void;
  onRemove: () => void;
  onCreateValue: (text: string) => Promise<number | null>;
  onRenameSection: (name: string) => Promise<void>;
  onRenameValue: (valueId: number, text: string) => Promise<void>;
  dragHandleProps?: React.HTMLAttributes<HTMLButtonElement>;
}

function resolveSelectedIds(section: PromptTemplateSection): number[] {
  if (section.selected_value_ids?.length) return section.selected_value_ids;
  if (section.selected_value_id != null) return [section.selected_value_id];
  return [];
}

function orderedSelectedIds(section: PromptTemplateSection, ids: number[]): number[] {
  const selected = new Set(ids);
  return section.values.filter((value) => selected.has(value.id)).map((value) => value.id);
}

const modeButtonClassName =
  "rounded-md px-2 py-1 text-xs font-medium border transition-colors disabled:opacity-50 disabled:pointer-events-none";

export default function SectionSelectionRow({
  section,
  onChange,
  onRemove,
  onCreateValue,
  onRenameSection,
  onRenameValue,
  dragHandleProps,
}: SectionSelectionRowProps) {
  const [createValueOpen, setCreateValueOpen] = useState(false);
  const [newValueText, setNewValueText] = useState("");
  const [creatingValue, setCreatingValue] = useState(false);
  const [pickKey, setPickKey] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const isRandom = section.selection_mode === "random";

  const selectedIds = resolveSelectedIds(section);
  const selectedIdSet = new Set(selectedIds);
  const selectedValues = section.values.filter((value) => selectedIdSet.has(value.id));
  const unselectedValues = section.values.filter((value) => !selectedIdSet.has(value.id));

  function updateSelection(ids: number[]) {
    onChange({
      selected_value_ids: orderedSelectedIds(section, ids),
    });
  }

  function defaultSelectedIds(): number[] {
    return section.values.length ? [section.values[0].id] : [];
  }

  async function handleCreateValue() {
    const text = newValueText.trim();
    if (!text) return;
    setCreatingValue(true);
    try {
      const valueId = await onCreateValue(text);
      setCreateValueOpen(false);
      setNewValueText("");
      if (valueId != null) {
        updateSelection([...selectedIds, valueId]);
      }
    } finally {
      setCreatingValue(false);
    }
  }

  function handleValuePick(raw: string) {
    if (raw === CREATE_OPTION) {
      setCreateValueOpen(true);
      return;
    }
    if (!raw) return;
    const valueId = Number(raw);
    if (selectedIdSet.has(valueId)) return;
    updateSelection([...selectedIds, valueId]);
    setPickKey((key) => key + 1);
  }

  function removeSelectedValue(valueId: number) {
    updateSelection(selectedIds.filter((id) => id !== valueId));
  }

  function addAllValues() {
    updateSelection(section.values.map((value) => value.id));
  }

  const collapsedSummary =
    selectedValues.length > 0
      ? selectedValues.map((value) => value.text).join(", ")
      : isRandom
        ? "No values in pool"
        : "No values selected";

  return (
    <>
      <div className="rounded-lg border border-border bg-surface/40 p-3 space-y-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="text-muted hover:text-text cursor-grab active:cursor-grabbing p-1"
            aria-label={`Reorder ${section.section_name}`}
            {...dragHandleProps}
          >
            <GripVertical size={16} />
          </button>
          <button
            type="button"
            onClick={() => setExpanded((open) => !open)}
            className="text-muted hover:text-text p-1"
            aria-expanded={expanded}
            aria-label={expanded ? `Collapse ${section.section_name}` : `Expand ${section.section_name}`}
          >
            <ChevronDown
              size={16}
              className={cn("transition-transform", expanded ? "rotate-0" : "-rotate-90")}
            />
          </button>
          <div className="flex-1 min-w-0">
            <InlineRenameField
              value={section.section_name}
              onSave={onRenameSection}
              ariaLabel={`Rename section ${section.section_name}`}
              className="block w-full text-sm font-medium text-text"
              inputClassName="text-sm font-medium"
            />
            {!expanded && (
              <div className="text-[11px] text-muted truncate">
                {isRandom ? "Random" : "Fixed"}
                {isRandom && section.include_empty ? " · +empty" : ""} · {collapsedSummary}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onRemove}
            className="text-muted hover:text-error p-1"
            aria-label={`Remove ${section.section_name}`}
          >
            <Trash2 size={16} />
          </button>
        </div>

        {expanded && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className={cn(
                  modeButtonClassName,
                  !isRandom
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border text-muted hover:text-text",
                )}
                onClick={() =>
                  onChange({
                    selection_mode: "fixed",
                    selected_value_ids: selectedIds.length
                      ? orderedSelectedIds(section, selectedIds)
                      : defaultSelectedIds(),
                  })
                }
              >
                Fixed
              </button>
              <button
                type="button"
                className={cn(
                  modeButtonClassName,
                  isRandom
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border text-muted hover:text-text",
                )}
                onClick={() =>
                  onChange({
                    selection_mode: "random",
                    selected_value_ids: selectedIds.length
                      ? orderedSelectedIds(section, selectedIds)
                      : section.values.map((value) => value.id),
                  })
                }
              >
                Random
              </button>
              <button
                type="button"
                className={cn(modeButtonClassName, "border-border text-muted hover:text-text")}
                onClick={addAllValues}
                disabled={!section.values.length || unselectedValues.length === 0}
              >
                Add all
              </button>
              {isRandom && (
                <label className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-muted cursor-pointer hover:text-text">
                  <input
                    type="checkbox"
                    className="rounded border-border"
                    checked={section.include_empty}
                    onChange={(event) => onChange({ include_empty: event.target.checked })}
                  />
                  Include empty
                </label>
              )}
            </div>

            {selectedValues.length ? (
              <div className="flex flex-wrap gap-1.5">
                {selectedValues.map((value) => (
                  <span
                    key={value.id}
                    className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-surface/60 px-1.5 py-0.5"
                  >
                    <InlineRenameField
                      value={value.text}
                      onSave={(text) => onRenameValue(value.id, text)}
                      ariaLabel={`Rename value ${value.text}`}
                      className="truncate font-mono text-[11px] text-text"
                      inputClassName="font-mono text-[11px]"
                    />
                    <button
                      type="button"
                      onClick={() => removeSelectedValue(value.id)}
                      className="shrink-0 text-muted hover:text-error"
                      aria-label={`Remove ${value.text}`}
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted">
                {isRandom
                  ? "No values in random pool. Add one from the picker below."
                  : "No values selected. Add one from the picker below."}
              </p>
            )}

            <select
              key={pickKey}
              className={selectClassName}
              value=""
              onChange={(event) => handleValuePick(event.target.value)}
            >
              <option value="" disabled>
                Add value…
              </option>
              {unselectedValues.map((value) => (
                <option key={value.id} value={value.id}>
                  {value.text}
                </option>
              ))}
              <option value={CREATE_OPTION}>+ Create new value…</option>
            </select>
          </>
        )}
      </div>

      <Modal
        open={createValueOpen}
        onClose={() => {
          setCreateValueOpen(false);
          setNewValueText("");
        }}
        title="Create value"
        description={`Add a value to "${section.section_name}".`}
        size="sm"
      >
        <Input
          label="Value text"
          placeholder="e.g. 1girl, masterpiece"
          value={newValueText}
          onChange={(event) => setNewValueText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleCreateValue();
          }}
          autoFocus
        />
        <ModalFooter>
          <Button
            variant="secondary"
            onClick={() => {
              setCreateValueOpen(false);
              setNewValueText("");
            }}
          >
            Cancel
          </Button>
          <Button onClick={() => void handleCreateValue()} disabled={!newValueText.trim() || creatingValue}>
            Create
          </Button>
        </ModalFooter>
      </Modal>
    </>
  );
}
