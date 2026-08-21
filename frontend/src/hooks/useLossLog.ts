"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { lorasApi } from "@/lib/api/loras";
import { mergeLossSeries } from "@/lib/lossUtils";
import type { LossPoint } from "@/types";

type SeriesMap = Record<string, LossPoint[]>;

export default function useLossLog(
  loraId: number,
  reloadInterval: number | null = null,
  resetKey: string | null = null,
) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error" | "refreshing">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});
  const keysRef = useRef<string[]>([]);

  const lossKeys = useMemo(() => {
    if (keys.length === 0) return ["loss/loss"];
    return [...keys].sort();
  }, [keys]);

  const refreshLoss = useCallback(async () => {
    if (!loraId || inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(didInitialLoadRef.current ? "refreshing" : "loading");
    setErrorMessage(null);

    try {
      const isInitialLoad = !didInitialLoadRef.current;
      const currentKeys = keysRef.current;
      const wantedKeys = isInitialLoad
        ? undefined
        : (currentKeys.length ? [...currentKeys] : ["loss/loss"]).sort();

      const batch = await lorasApi.getLossBatch(loraId, {
        keys: wantedKeys,
        sinceSteps: wantedKeys?.map((key) => lastStepByKeyRef.current[key] ?? null),
      });

      const discoveredKeys = batch.keys.length ? batch.keys : wantedKeys ?? ["loss/loss"];
      keysRef.current = discoveredKeys;
      setKeys(discoveredKeys);

      const resolvedKeys = (discoveredKeys.length ? discoveredKeys : ["loss/loss"]).sort();
      const results = resolvedKeys.map((key) => ({
        key,
        keys: discoveredKeys,
        points: batch.series[key] ?? [],
      }));

      setSeries((previous) => {
        const merged = mergeLossSeries({
          previous,
          results,
          wantedKeys: resolvedKeys,
          isInitialLoad,
          lastStepByKey: lastStepByKeyRef.current,
        });
        lastStepByKeyRef.current = merged.lastStepByKey;
        return merged.next;
      });

      setStatus("success");
      didInitialLoadRef.current = true;
    } catch (error) {
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Failed to load loss logs");
    } finally {
      inFlightRef.current = false;
    }
  }, [loraId]);

  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    keysRef.current = [];
    setSeries({});
    setKeys([]);
    setStatus("idle");
    setErrorMessage(null);
    void refreshLoss();

    if (reloadInterval) {
      const interval = setInterval(() => void refreshLoss(), reloadInterval);
      return () => clearInterval(interval);
    }
  }, [loraId, reloadInterval, resetKey, refreshLoss]);

  return { series, keys, lossKeys, status, errorMessage, refreshLoss };
}

export type { LossPoint };
