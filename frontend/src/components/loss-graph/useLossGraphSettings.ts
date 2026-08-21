"use client";

import { useEffect, useRef, useState } from "react";

export interface PersistedLossGraphSettings {
  useLogScale: boolean;
  showTrend: boolean;
  smoothing: number;
  plotStride: number;
  clipOutliers: boolean;
  enabled: Record<string, boolean>;
}

function settingsStorageKey(): string | null {
  if (typeof window === "undefined") return null;
  return `lossGraph:${window.location.pathname}${window.location.search}`;
}

export function useLossGraphSettings(loraId: number, lossKeys: string[]) {
  const [useLogScale, setUseLogScale] = useState(false);
  const [showTrend, setShowTrend] = useState(true);
  const [smoothing, setSmoothing] = useState(80);
  const [plotStride, setPlotStride] = useState(1);
  const [clipOutliers, setClipOutliers] = useState(false);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [hydrated, setHydrated] = useState(false);
  const persistedEnabledRef = useRef<Record<string, boolean> | null>(null);

  useEffect(() => {
    setHydrated(false);
    persistedEnabledRef.current = null;
    const key = settingsStorageKey();
    if (!key) {
      setHydrated(true);
      return;
    }
    try {
      const raw = localStorage.getItem(key);
      if (raw) {
        const s = JSON.parse(raw) as Partial<PersistedLossGraphSettings>;
        if (typeof s.useLogScale === "boolean") setUseLogScale(s.useLogScale);
        if (typeof s.showTrend === "boolean") setShowTrend(s.showTrend);
        if (typeof s.smoothing === "number") setSmoothing(s.smoothing);
        if (typeof s.plotStride === "number") setPlotStride(s.plotStride);
        if (typeof s.clipOutliers === "boolean") setClipOutliers(s.clipOutliers);
        if (s.enabled && typeof s.enabled === "object") {
          persistedEnabledRef.current = s.enabled;
          setEnabled(s.enabled);
        }
      }
    } catch {
      // ignore
    }
    setHydrated(true);
  }, [loraId]);

  useEffect(() => {
    if (!hydrated) return;
    const key = settingsStorageKey();
    if (!key) return;
    try {
      const payload: PersistedLossGraphSettings = {
        useLogScale,
        showTrend,
        smoothing,
        plotStride,
        clipOutliers,
        enabled,
      };
      localStorage.setItem(key, JSON.stringify(payload));
    } catch {
      // ignore
    }
  }, [hydrated, useLogScale, showTrend, smoothing, plotStride, clipOutliers, enabled]);

  useEffect(() => {
    if (lossKeys.length === 0) return;
    setEnabled((prev) => {
      const next = { ...prev };
      for (const k of lossKeys) {
        if (next[k] === undefined) next[k] = persistedEnabledRef.current?.[k] ?? k === "loss/loss";
      }
      for (const k of Object.keys(next)) {
        if (!lossKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [lossKeys]);

  return {
    useLogScale,
    setUseLogScale,
    showTrend,
    setShowTrend,
    smoothing,
    setSmoothing,
    plotStride,
    setPlotStride,
    clipOutliers,
    setClipOutliers,
    enabled,
    setEnabled,
  };
}
