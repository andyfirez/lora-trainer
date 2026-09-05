"use client";

import useSWR from "swr";
import { samplingsApi } from "@/lib/api/samplings";
import { samplerOptions, schedulerOptions, type SelectOption } from "@/lib/sampleSamplerOptions";
import type { SamplingProfile } from "@/types";

export function useSamplingOptions(profile: SamplingProfile = "comfy") {
  const { data, error, isLoading } = useSWR(["/samplings/options", profile], () =>
    samplingsApi.getOptions(profile),
  );
  const samplers: SelectOption[] =
    data?.samplers.map((item) => ({
      value: item.id,
      label: item.label,
      defaultScheduler: item.default_scheduler,
    })) ?? samplerOptions;
  const schedulers: SelectOption[] =
    data?.schedulers.map((item) => ({
      value: item.id,
      label: item.label,
    })) ?? schedulerOptions;
  return { samplers, schedulers, profile: data?.profile ?? profile, isLoading, error };
}
