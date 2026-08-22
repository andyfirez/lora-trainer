import type { PromptSelectionMode, PromptTemplateSection } from "@/types/promptBuilder";

export const PROMPT_SEGMENT_SEPARATOR = ", ";

function composeFromSelectedTexts(texts: string[]): string {
  const cleaned = texts.map((text) => text.trim()).filter(Boolean);
  if (!cleaned.length) return "";
  if (cleaned.length === 1) return cleaned[0];
  return `{${cleaned.join("|")}}`;
}

function composeFixedTexts(texts: string[]): string {
  return texts
    .map((text) => text.trim())
    .filter(Boolean)
    .join(PROMPT_SEGMENT_SEPARATOR);
}

function composeRandomTexts(texts: string[], includeEmpty = false): string {
  const cleaned = texts.map((text) => text.trim()).filter(Boolean);
  if (includeEmpty) {
    if (!cleaned.length) return "{|}";
    return `{|${cleaned.join("|")}}`;
  }
  if (!cleaned.length) return "";
  if (cleaned.length === 1) return cleaned[0];
  return `{${cleaned.join("|")}}`;
}

function resolveSelectedIds(section: PromptTemplateSection): number[] {
  if (section.selected_value_ids?.length) return section.selected_value_ids;
  if (section.selected_value_id != null) return [section.selected_value_id];
  return [];
}

export function composeSection(
  values: string[],
  mode: PromptSelectionMode,
  selectedTexts: string[],
): string {
  const cleaned = values.map((value) => value.trim()).filter(Boolean);
  if (mode === "random") {
    return composeRandomTexts(cleaned);
  }
  return composeFixedTexts(selectedTexts);
}

export function composePrompt(segments: string[], separator = PROMPT_SEGMENT_SEPARATOR): string {
  return segments.filter((segment) => segment.trim()).join(separator);
}

export function composeTemplatePrompt(sections: PromptTemplateSection[]): string {
  const segments = sections
    .slice()
    .sort((a, b) => a.sort_order - b.sort_order)
    .map((section) => {
      const selectedIds = resolveSelectedIds(section);
      const texts =
        section.selection_mode === "random" && !selectedIds.length
          ? section.values.map((value) => value.text)
          : section.values
              .filter((value) => new Set(selectedIds).has(value.id))
              .map((value) => value.text);
      return section.selection_mode === "random"
        ? composeRandomTexts(texts, section.include_empty)
        : composeFixedTexts(texts);
    });
  return composePrompt(segments);
}
