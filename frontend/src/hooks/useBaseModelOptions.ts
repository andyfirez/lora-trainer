"use client";

import useSWR from "swr";
import { buildBaseModelSelectOptions } from "@/lib/baseModelOptions";
import { storageApi } from "@/lib/api/storage";

export { buildBaseModelSelectOptions } from "@/lib/baseModelOptions";

export function useBaseModelOptions(currentValues: string[] = []) {
  const { data, error, isLoading } = useSWR("/storage/base-models", () => storageApi.listBaseModels());

  const options = buildBaseModelSelectOptions(data?.models ?? [], currentValues);

  if (!isLoading && options.length === 0) {
    return {
      options: [{ value: "", label: "No models found in base models folder", disabled: true }],
      root: data?.root ?? null,
      isLoading,
      error,
    };
  }

  return {
    options,
    root: data?.root ?? null,
    isLoading,
    error,
  };
}
