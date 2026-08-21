"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import {
  computeCanvasSize,
  FALLBACK_CANVAS_HEIGHT,
  MIN_CANVAS_HEIGHT,
  type BuiltChartData,
} from "./lossGraphMath";

export interface LossGraphCanvasProps {
  built: BuiltChartData;
  hasData: boolean;
  status: "idle" | "loading" | "success" | "error" | "refreshing";
  activeKeys: string[];
  showTrend: boolean;
  useLogScale: boolean;
}

export default function LossGraphCanvas({
  built,
  hasData,
  status,
  activeKeys,
  showTrend,
  useLogScale,
}: LossGraphCanvasProps) {
  const [isZoomed, setIsZoomed] = useState(false);
  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);
  const isZoomedRef = useRef(false);

  useEffect(() => {
    isZoomedRef.current = isZoomed;
  }, [isZoomed]);

  const structuralKey = `${activeKeys.join("|")}|trend=${showTrend}|log=${useLogScale}|has=${hasData}`;

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!containerRef.current || !chartHostRef.current || !hasData) return;

    const host = chartHostRef.current;
    const rect = host.getBoundingClientRect();
    const initialHeight = rect.height > 0 ? Math.max(MIN_CANVAS_HEIGHT, rect.height - 40) : FALLBACK_CANVAS_HEIGHT;
    const opts: uPlot.Options = {
      width: rect.width || 800,
      height: initialHeight,
      padding: [12, 16, 0, 4],
      series: built.seriesConfigs,
      scales: built.scales,
      axes: built.axes,
      cursor: { drag: { x: true, y: false, setScale: true }, points: { size: 6 } },
      legend: { show: true },
      hooks: {
        setScale: [
          (u, key) => {
            if (key !== "x") return;
            const xs = u.data[0] as number[];
            if (!xs?.length) return;
            const sx = u.scales.x;
            const zoomed = sx.min !== xs[0] || sx.max !== xs[xs.length - 1];
            setIsZoomed(zoomed);
          },
        ],
      },
    };

    uplotRef.current = new uPlot(opts, built.data, containerRef.current);
    setIsZoomed(false);

    const raf = requestAnimationFrame(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(host);
      if (fitted) u.setSize(fitted);
    });

    return () => {
      cancelAnimationFrame(raf);
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
  }, [structuralKey, hasData, built.axes, built.data, built.scales, built.seriesConfigs]);

  useEffect(() => {
    const u = uplotRef.current;
    if (!u) return;
    if (isZoomedRef.current) {
      u.setData(built.data, false);
      u.redraw(true, true);
    } else {
      u.setData(built.data, true);
    }
  }, [built]);

  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(el);
      if (fitted) u.setSize(fitted);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  const handleResetZoom = useCallback(() => {
    const u = uplotRef.current;
    if (!u) return;
    const xs = u.data[0] as number[];
    if (!xs?.length) return;
    u.setScale("x", { min: xs[0], max: xs[xs.length - 1] });
  }, []);

  return (
    <>
      <div className="px-4 pt-4 pb-4 flex flex-col">
        <div
          className="bg-black/30 rounded-lg border border-border relative select-none"
          style={{ minHeight: 280, height: 320 }}
        >
          {!hasData ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">
              {status === "error" ? "Failed to load loss logs." : "Waiting for loss points…"}
            </div>
          ) : (
            <>
              {isZoomed && (
                <button
                  type="button"
                  onClick={handleResetZoom}
                  className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs bg-accent/80 hover:bg-accent text-white border border-accent/50"
                >
                  Reset zoom
                </button>
              )}
              <div ref={chartHostRef} className="absolute top-0 left-0 right-0 bottom-2 overflow-hidden">
                <div ref={containerRef} />
              </div>
            </>
          )}
        </div>
      </div>

      <style jsx global>{`
        .uplot,
        .uplot * {
          font-family: inherit;
        }
        .uplot .u-legend {
          color: rgba(255, 255, 255, 0.85);
          font-size: 12px;
          margin-top: 4px;
        }
        .uplot .u-legend th,
        .uplot .u-legend td {
          color: rgba(255, 255, 255, 0.85);
        }
        .uplot .u-select {
          background: rgba(217, 119, 6, 0.15);
          border: 1px solid rgba(217, 119, 6, 0.4);
        }
      `}</style>
    </>
  );
}
