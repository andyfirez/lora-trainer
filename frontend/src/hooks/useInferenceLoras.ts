"use client";

import useSWR from "swr";
import { storageApi } from "@/lib/api/storage";

export function useInferenceLoras() {
  return useSWR("/storage/inference-loras", () => storageApi.listInferenceLoras());
}
