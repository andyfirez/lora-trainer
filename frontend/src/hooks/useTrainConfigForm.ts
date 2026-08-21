"use client";

import { useEffect, useMemo } from "react";
import useSWR from "swr";
import { datasetsApi } from "@/lib/api/datasets";
import type { OptimizerType } from "@/lib/optimizerPresets";
import {
  applyTrainConfigNestedPatch,
  applyTrainConfigPatch,
  applyTrainOptimizerType,
  datasetOptionLabel,
  getOptimizerType,
  isDatasetCompatibleWithTrain,
  isTextEncoderTrainingEnabled,
  sanitizeTrainConfig,
  type TrainConfig,
} from "@/lib/trainConfigSanitize";
import type { Dataset } from "@/types";

export interface TrainConfigFormContext {
  config: TrainConfig;
  datasets: Dataset[] | undefined;
  datasetsLoading: boolean;
  concepts: TrainConfig[];
  set: (key: string, value: unknown) => void;
  setNested: (parent: string, key: string, value: unknown) => void;
  setOptimizerType: (type: OptimizerType) => void;
  updateConcept: (index: number, key: string, value: unknown) => void;
  addConcept: () => void;
  removeConcept: (index: number) => void;
  datasetById: (id: number | undefined) => Dataset | undefined;
  isDatasetCompatible: (dataset: Dataset) => boolean;
  datasetOptions: { value: string; label: string; disabled?: boolean }[];
  trainResolution: number;
  trainEnableBucket: boolean;
  optimizerType: OptimizerType;
  isAdamFamily: boolean;
  isAdafactor: boolean;
  isProdigy: boolean;
  checkpointingEnabled: boolean;
  cacheLatentsEnabled: boolean;
  textEncoderTrainingEnabled: boolean;
  cacheTextEncoderEnabled: boolean;
}

export function useTrainConfigForm(
  config: TrainConfig,
  onChange: (config: TrainConfig) => void,
): TrainConfigFormContext {
  const concepts: TrainConfig[] = (config.concepts as TrainConfig[] | undefined) ?? [];
  const { data: datasets, isLoading: datasetsLoading } = useSWR("/datasets", () => datasetsApi.list());

  function emit(next: TrainConfig) {
    onChange(sanitizeTrainConfig(next, datasets));
  }

  function set(key: string, value: unknown) {
    emit(applyTrainConfigPatch(config, key, value, datasets));
  }

  function setNested(parent: string, key: string, value: unknown) {
    emit(applyTrainConfigNestedPatch(config, parent, key, value, datasets));
  }

  function setOptimizerType(type: OptimizerType) {
    emit(applyTrainOptimizerType(config, type, datasets));
  }

  const optimizerType = getOptimizerType(config);
  const isAdamFamily = optimizerType === "adamw" || optimizerType === "adamw_8bit";
  const isAdafactor = optimizerType === "adafactor";
  const isProdigy = optimizerType === "prodigy";

  const trainResolution = Number(config.resolution ?? 1024);
  const trainEnableBucket = Boolean(config.enable_bucket);

  function isDatasetCompatible(dataset: Dataset): boolean {
    return isDatasetCompatibleWithTrain(dataset, trainResolution, trainEnableBucket);
  }

  const datasetOptions = useMemo(
    () =>
      (datasets ?? []).map((dataset) => ({
        value: String(dataset.id),
        label: datasetOptionLabel(dataset, trainResolution, trainEnableBucket),
        disabled: !isDatasetCompatible(dataset),
      })),
    [datasets, trainResolution, trainEnableBucket],
  );

  function datasetById(id: number | undefined): Dataset | undefined {
    if (id == null) return undefined;
    return datasets?.find((dataset) => dataset.id === id);
  }

  function updateConcept(index: number, key: string, value: unknown) {
    const next = concepts.map((concept, idx) => (idx === index ? { ...concept, [key]: value } : concept));
    set("concepts", next);
  }

  function addConcept() {
    const compatible = datasets?.find((dataset) => isDatasetCompatible(dataset));
    const defaultDatasetId = compatible?.id ?? datasets?.[0]?.id;
    if (defaultDatasetId == null) return;
    set("concepts", [
      ...concepts,
      { dataset_id: defaultDatasetId, trigger_words: [], caption_extension: ".txt", repeats: 1 },
    ]);
  }

  function removeConcept(index: number) {
    set(
      "concepts",
      concepts.filter((_, idx) => idx !== index),
    );
  }

  const checkpointingEnabled = (config.checkpointing_enabled as boolean | undefined) ?? true;
  const cacheLatentsEnabled = (config.cache_latents as boolean | undefined) ?? true;
  const textEncoderTrainingEnabled = isTextEncoderTrainingEnabled(config);
  const cacheTextEncoderEnabled = textEncoderTrainingEnabled
    ? false
    : ((config.cache_text_encoder_outputs as boolean | undefined) ?? true);

  useEffect(() => {
    if (datasetsLoading || !datasets?.length) return;
    const normalized = sanitizeTrainConfig(config, datasets);
    const before = JSON.stringify({
      concepts: config.concepts ?? [],
      clip_skip: config.clip_skip ?? null,
      cache_text_encoder_outputs: config.cache_text_encoder_outputs ?? null,
      cache_text_encoder_outputs_to_disk: config.cache_text_encoder_outputs_to_disk ?? null,
    });
    const after = JSON.stringify({
      concepts: normalized.concepts ?? [],
      clip_skip: normalized.clip_skip ?? null,
      cache_text_encoder_outputs: normalized.cache_text_encoder_outputs ?? null,
      cache_text_encoder_outputs_to_disk: normalized.cache_text_encoder_outputs_to_disk ?? null,
    });
    if (before !== after) {
      onChange(normalized);
    }
  }, [config, datasets, datasetsLoading, onChange]);

  return {
    config,
    datasets,
    datasetsLoading,
    concepts,
    set,
    setNested,
    setOptimizerType,
    updateConcept,
    addConcept,
    removeConcept,
    datasetById,
    isDatasetCompatible,
    datasetOptions,
    trainResolution,
    trainEnableBucket,
    optimizerType,
    isAdamFamily,
    isAdafactor,
    isProdigy,
    checkpointingEnabled,
    cacheLatentsEnabled,
    textEncoderTrainingEnabled,
    cacheTextEncoderEnabled,
  };
}
