"use client";

import useSWR from "swr";
import type { RunnableResponse } from "@/types";

export function useRunnableDetail<T extends RunnableResponse>(swrKey: string, fetcher: () => Promise<T>) {
  return useSWR(swrKey, fetcher, {
    refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 2000),
  });
}
