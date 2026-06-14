"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { jobsApi, type LossPoint, type JobLossResponse } from "@/lib/api/jobs";

type SeriesMap = Record<string, LossPoint[]>;

export default function useJobLossLog(
  jobId: number,
  reloadInterval: number | null = null,
  resetKey: string | null = null,
) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error" | "refreshing">("idle");

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const lossKeys = useMemo(() => {
    if (keys.length === 0) return ["loss/loss"];
    return [...keys].sort();
  }, [keys]);

  const refreshLoss = useCallback(async () => {
    if (!jobId || inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(didInitialLoadRef.current ? "refreshing" : "loading");

    try {
      const first = await jobsApi.getLoss(jobId, { key: "loss/loss", limit: 1 });
      const newKeys = first.keys ?? [];
      setKeys(newKeys);

      const wantedKeys = (newKeys.length ? [...newKeys] : ["loss/loss"]).sort();
      const results = await Promise.all(
        wantedKeys.map(async (k) => {
          const params: { key: string; limit: number; since_step?: number } = { key: k, limit: 1_000_000 };
          if (reloadInterval && lastStepByKeyRef.current[k] != null) {
            params.since_step = lastStepByKeyRef.current[k]!;
          }
          return jobsApi.getLoss(jobId, params);
        }),
      );

      setSeries((prev) => {
        const next: SeriesMap = { ...prev };
        for (const r of results) {
          const k = r.key;
          const newPoints = (r.points ?? []).filter((p) => p.value !== null);

          if (!didInitialLoadRef.current) {
            next[k] = newPoints;
          } else if (newPoints.length) {
            const existing = next[k] ?? [];
            const prevLast = existing.length ? existing[existing.length - 1].step : null;
            const filtered = prevLast == null ? newPoints : newPoints.filter((p) => p.step > prevLast);
            next[k] = filtered.length ? [...existing, ...filtered] : existing;
          } else {
            next[k] = next[k] ?? [];
          }

          const finalArr = next[k] ?? [];
          lastStepByKeyRef.current[k] = finalArr.length
            ? finalArr[finalArr.length - 1].step
            : (lastStepByKeyRef.current[k] ?? null);
        }

        for (const existingKey of Object.keys(next)) {
          if (!wantedKeys.includes(existingKey)) {
            delete next[existingKey];
            delete lastStepByKeyRef.current[existingKey];
          }
        }
        return next;
      });

      setStatus("success");
      didInitialLoadRef.current = true;
    } catch {
      setStatus("error");
    } finally {
      inFlightRef.current = false;
    }
  }, [jobId, reloadInterval]);

  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus("idle");
    void refreshLoss();

    if (reloadInterval) {
      const interval = setInterval(() => void refreshLoss(), reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobId, reloadInterval, resetKey, refreshLoss]);

  return { series, keys, lossKeys, status, refreshLoss };
}

export type { LossPoint, JobLossResponse };
