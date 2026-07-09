export type ParameterMeta = {
  key: string;
  label: string;
  section: string;
  shortHint: string;
  description: string;
  defaultValue?: string;
  constraints?: string;
  yamlOnly?: boolean;
  deprecated?: boolean;
};

export const TRAIN_SECTION_ORDER = [
  "Model",
  "LoRA",
  "Training Targets",
  "Training",
  "Optimizer",
  "Data",
  "Optimization",
  "Performance",
  "Checkpointing",
  "Sampling",
  "Logging",
  "Advanced",
] as const;

export function parameterAnchor(key: string): string {
  return key.replace(/\./g, "-").replace(/_/g, "-");
}

export function buildParameterLookup(
  entries: ParameterMeta[],
): Map<string, ParameterMeta> {
  return new Map(entries.map((entry) => [entry.key, entry]));
}

export function groupBySection(
  entries: ParameterMeta[],
  sectionOrder: readonly string[],
): { section: string; items: ParameterMeta[] }[] {
  const grouped = new Map<string, ParameterMeta[]>();
  for (const entry of entries) {
    const list = grouped.get(entry.section) ?? [];
    list.push(entry);
    grouped.set(entry.section, list);
  }
  const ordered = sectionOrder
    .filter((section) => grouped.has(section))
    .map((section) => ({ section, items: grouped.get(section)! }));
  const extra = [...grouped.keys()]
    .filter((section) => !sectionOrder.includes(section))
    .sort()
    .map((section) => ({ section, items: grouped.get(section)! }));
  return [...ordered, ...extra];
}
