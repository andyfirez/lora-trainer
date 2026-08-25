export interface BaseModelSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export function buildBaseModelSelectOptions(
  models: { relative_path: string; is_dir: boolean }[],
): BaseModelSelectOption[] {
  return models.map((model) => ({
    value: model.relative_path,
    label: model.relative_path,
  }));
}
