"use client";

import { useEffect, useState } from "react";
import { useDebouncedCallback } from "@/hooks/useDebouncedCallback";
import { samplingsApi } from "@/lib/api/samplings";
import type { PromptTokenCountResponse } from "@/types";

export function usePromptTokenCount(prompt: string, baseModelName: string): PromptTokenCountResponse | null {
  const [counts, setCounts] = useState<PromptTokenCountResponse | null>(null);

  const request = useDebouncedCallback((text: string, model: string) => {
    if (!model.trim()) {
      setCounts(null);
      return;
    }
    void samplingsApi
      .tokenCount(text, model)
      .then(setCounts)
      .catch(() => setCounts(null));
  }, 300);

  useEffect(() => {
    request(prompt, baseModelName);
  }, [prompt, baseModelName, request]);

  return counts;
}
