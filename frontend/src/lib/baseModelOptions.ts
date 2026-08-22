export interface BaseModelSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export function buildBaseModelSelectOptions(
  models: { relative_path: string; is_dir: boolean }[],
  currentValues: string[] = [],
): BaseModelSelectOption[] {
  const options: BaseModelSelectOption[] = models.map((model) => ({
    value: model.relative_path,
    label: model.relative_path,
  }));

  for (const value of currentValues) {
    const trimmed = value.trim();
    if (!trimmed || options.some((option) => option.value === trimmed)) {
      continue;
    }
    options.unshift({
      value: trimmed,
      label: `${trimmed} (not in base models folder)`,
    });
  }

  return options;
}
