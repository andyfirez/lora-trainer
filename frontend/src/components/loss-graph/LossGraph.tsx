"use client";

import useLossLog from "@/hooks/useLossLog";
import { useEffect, useMemo, useRef } from "react";
import LossGraphCanvas from "./LossGraphCanvas";
import LossGraphControls from "./LossGraphControls";
import { buildChartData } from "./lossGraphMath";
import { useLossGraphSettings } from "./useLossGraphSettings";

interface Props {
  loraId: number;
  isActive: boolean;
  resetKey: string | null;
}

export default function LossGraph({ loraId, isActive, resetKey }: Props) {
  const { series, lossKeys, status, refreshLoss } = useLossLog(
    loraId,
    isActive ? 2000 : null,
    resetKey,
  );

  const {
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
  } = useLossGraphSettings(loraId, lossKeys);

  const activeKeys = useMemo(() => lossKeys.filter((k) => enabled[k] !== false), [lossKeys, enabled]);

  const yClipRef = useRef<Record<string, { min: number; max: number }> | null>(null);

  const built = useMemo(
    () =>
      buildChartData({
        series,
        activeKeys,
        smoothing,
        plotStride,
        useLogScale,
        showTrend,
        clipOutliers,
        yClipRef,
      }),
    [series, activeKeys, smoothing, plotStride, useLogScale, showTrend, clipOutliers],
  );

  useEffect(() => {
    yClipRef.current = built.yClip;
  }, [built.yClip]);

  const hasData = (built.data[0]?.length ?? 0) > 1;
  const totalPoints = built.data[0]?.length ?? 0;

  return (
    <div className="bg-surface rounded-xl border border-border overflow-hidden flex flex-col">
      <div className="px-4 py-3 flex items-center justify-between border-b border-border">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-accent" />
          <h2 className="text-sm font-medium text-text">Loss Graph</h2>
          <span className="text-xs text-muted">
            {status === "loading" && "Loading…"}
            {status === "refreshing" && "Refreshing…"}
            {status === "error" && "Error"}
            {status === "success" && hasData && `${totalPoints.toLocaleString()} steps`}
            {status === "success" && !hasData && "No data yet"}
          </span>
        </div>
        <button
          type="button"
          onClick={() => void refreshLoss()}
          className="px-3 py-1 rounded-md text-xs bg-white/5 hover:bg-white/10 text-muted hover:text-text border border-border"
        >
          Refresh
        </button>
      </div>

      <LossGraphCanvas
        built={built}
        hasData={hasData}
        status={status}
        activeKeys={activeKeys}
        showTrend={showTrend}
        useLogScale={useLogScale}
      />

      <LossGraphControls
        lossKeys={lossKeys}
        showTrend={showTrend}
        useLogScale={useLogScale}
        clipOutliers={clipOutliers}
        smoothing={smoothing}
        plotStride={plotStride}
        enabled={enabled}
        onToggleTrend={() => setShowTrend((v) => !v)}
        onToggleLogScale={() => setUseLogScale((v) => !v)}
        onToggleClipOutliers={() => setClipOutliers((v) => !v)}
        onSmoothingChange={setSmoothing}
        onPlotStrideChange={setPlotStride}
        onToggleSeries={(key) => setEnabled((prev) => ({ ...prev, [key]: !(prev[key] ?? true) }))}
      />
    </div>
  );
}
