export type ValueOption = {
  value: string;
  description: string;
};

export type RangeGuidance = {
  range: string;
  description: string;
};

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
  /** When false, no inline ? hint is shown on the config form. Defaults to true. */
  showInlineHint?: boolean;
  /** Community best-practice or default value shown on the reference page. */
  recommendedValue?: string;
  /** Per-value guidance for enum, boolean, and select parameters. */
  valueOptions?: ValueOption[];
  /** Practical numeric bands with effect notes for numeric parameters. */
  rangeGuidance?: RangeGuidance[];
};

export type TabGroup = {
  tab: string;
  label: string;
  sections: readonly string[];
};

export const TRAIN_TAB_GROUPS: TabGroup[] = [
  { tab: "setup", label: "Setup", sections: ["Model", "LoRA"] },
  { tab: "targets", label: "Targets", sections: ["Training Targets"] },
  { tab: "training", label: "Training", sections: ["Training", "Optimizer"] },
  { tab: "data-memory", label: "Data & Memory", sections: ["Data", "Optimization", "Performance"] },
  { tab: "output", label: "Output", sections: ["Checkpointing", "Sampling", "Logging"] },
  { tab: "advanced", label: "Advanced", sections: ["Advanced"] },
];

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

export function tabGroupForSection(section: string): string {
  const group = TRAIN_TAB_GROUPS.find((g) => g.sections.includes(section));
  return group?.tab ?? TRAIN_TAB_GROUPS[0].tab;
}

export function groupByTab(
  entries: ParameterMeta[],
  tabGroups: TabGroup[] = TRAIN_TAB_GROUPS,
): { tab: string; label: string; sections: { section: string; items: ParameterMeta[] }[] }[] {
  const bySection = groupBySection(entries, TRAIN_SECTION_ORDER);
  return tabGroups
    .map((group) => ({
      tab: group.tab,
      label: group.label,
      sections: group.sections
        .map((section) => bySection.find((s) => s.section === section))
        .filter((s): s is { section: string; items: ParameterMeta[] } => s !== undefined),
    }))
    .filter((group) => group.sections.length > 0);
}
