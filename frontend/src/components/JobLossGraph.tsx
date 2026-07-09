"use client";

import useJobLossLog, { type LossPoint } from "@/hooks/useJobLossLog";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface Props {
  jobId: number;
  isActive: boolean;
  resetKey: string | null;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return "";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(3);
  if (abs >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

const FALLBACK_CANVAS_HEIGHT = 360;
const MIN_CANVAS_HEIGHT = 160;

function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
  const { width, height } = host.getBoundingClientRect();
  if (width <= 0 || height <= 0) return null;
  const legend = host.querySelector(".u-legend") as HTMLElement | null;
  const legendH = legend?.getBoundingClientRect().height ?? 0;
  return { width, height: Math.max(MIN_CANVAS_HEIGHT, height - legendH) };
}

function emaPass(
  ys: (number | null)[],
  alpha: number,
  reverse: boolean,
): { vals: (number | null)[]; weights: number[] } {
  const vals: (number | null)[] = new Array(ys.length).fill(null);
  const weights: number[] = new Array(ys.length).fill(0);
  let s = 0;
  let n = 0;
  const start = reverse ? ys.length - 1 : 0;
  const step = reverse ? -1 : 1;
  for (let i = start; i >= 0 && i < ys.length; i += step) {
    const v = ys[i];
    if (v === null || !Number.isFinite(v)) continue;
    s = alpha * (v as number) + (1 - alpha) * s;
    n += 1;
    const w = 1 - Math.pow(1 - alpha, n);
    vals[i] = s / w;
    weights[i] = w;
  }
  return { vals, weights };
}

function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
  const fwd = emaPass(ys, alpha, false);
  const bwd = emaPass(ys, alpha, true);
  const out: (number | null)[] = new Array(ys.length);
  for (let i = 0; i < ys.length; i++) {
    const f = fwd.vals[i];
    const b = bwd.vals[i];
    if (f === null || b === null) {
      out[i] = null;
      continue;
    }
    const wf = fwd.weights[i];
    const wb = bwd.weights[i];
    const wsum = wf + wb;
    out[i] = wsum > 0 ? (wf * (f as number) + wb * (b as number)) / wsum : ((f as number) + (b as number)) / 2;
  }
  return out;
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

const PALETTE = [
  "rgba(212, 160, 84, 1)",
  "rgba(167, 139, 250, 1)",
  "rgba(74, 222, 128, 1)",
  "rgba(251, 191, 36, 1)",
  "rgba(244, 114, 182, 1)",
  "rgba(248, 113, 113, 1)",
  "rgba(34, 211, 238, 1)",
  "rgba(129, 140, 248, 1)",
];

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
}

function dulledColor(rgba: string): string {
  const m = rgba.match(/rgba?\((\d+),(\d+),(\d+)/);
  if (!m) return "rgba(120,120,120,1)";
  const r = Math.round(Number(m[1]) * 0.55);
  const g = Math.round(Number(m[2]) * 0.55);
  const b = Math.round(Number(m[3]) * 0.55);
  return `rgba(${r},${g},${b},1)`;
}

interface PersistedSettings {
  useLogScale: boolean;
  showTrend: boolean;
  smoothing: number;
  plotStride: number;
  clipOutliers: boolean;
  enabled: Record<string, boolean>;
}

function settingsStorageKey(): string | null {
  if (typeof window === "undefined") return null;
  return `jobLossGraph:${window.location.pathname}${window.location.search}`;
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "px-3 py-1 rounded-md text-xs border transition-colors",
        checked
          ? "bg-running/10 text-running border-running/30 hover:bg-running/15"
          : "bg-surface text-text-muted border-border hover:bg-white/5",
      ].join(" ")}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}

export default function JobLossGraph({ jobId, isActive, resetKey }: Props) {
  const { series, lossKeys, status, refreshLoss } = useJobLossLog(
    jobId,
    isActive ? 2000 : null,
    resetKey,
  );

  const [useLogScale, setUseLogScale] = useState(false);
  const [showTrend, setShowTrend] = useState(true);
  const [smoothing, setSmoothing] = useState(80);
  const [plotStride, setPlotStride] = useState(1);
  const [clipOutliers, setClipOutliers] = useState(false);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [isZoomed, setIsZoomed] = useState(false);
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
        const s = JSON.parse(raw) as Partial<PersistedSettings>;
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
  }, [jobId]);

  useEffect(() => {
    if (!hydrated) return;
    const key = settingsStorageKey();
    if (!key) return;
    try {
      const payload: PersistedSettings = { useLogScale, showTrend, smoothing, plotStride, clipOutliers, enabled };
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

  const activeKeys = useMemo(() => lossKeys.filter((k) => enabled[k] !== false), [lossKeys, enabled]);

  const yClipRef = useRef<Record<string, { min: number; max: number }> | null>(null);

  const built = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98;
    const fullAlpha = 0.005;

    const stepSet = new Set<number>();
    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        stepSet.add(p.step);
      }
    }
    let xs = Array.from(stepSet).sort((a, b) => a - b);
    if (stride > 1) xs = xs.filter((_, i) => i % stride === 0);
    const xsSet = new Set(xs);

    const data: (number[] | (number | null)[])[] = [xs];
    const seriesConfigs: uPlot.Series[] = [{}];
    const scales: uPlot.Scales = { x: { time: false } };
    const axes: uPlot.Axis[] = [
      {
        stroke: "rgba(255,255,255,0.55)",
        grid: { stroke: "rgba(255,255,255,0.06)" },
        ticks: { stroke: "rgba(255,255,255,0.15)" },
      },
    ];
    const scaleArrays: Record<string, (number | null)[][]> = {};

    for (let ki = 0; ki < activeKeys.length; ki++) {
      const key = activeKeys[ki];
      const scaleKey = `y::${key}`;
      const pts: LossPoint[] = series[key] ?? [];
      const map = new Map<number, number>();
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        if (!xsSet.has(p.step)) continue;
        map.set(p.step, p.value as number);
      }
      const raw: (number | null)[] = xs.map((s) => (map.has(s) ? (map.get(s) as number) : null));
      const smooth = emaWithNulls(raw, alpha);
      const fullSmooth = emaWithNulls(raw, fullAlpha);
      const color = strokeForKey(key);
      const colorDull = dulledColor(color);
      const colArrays: (number | null)[][] = [];

      data.push(smooth);
      seriesConfigs.push({
        label: key,
        scale: scaleKey,
        stroke: color,
        width: 2,
        spanGaps: false,
        points: { show: false },
        value: (_u, value) => formatNum(value),
      });
      colArrays.push(smooth);

      if (showTrend) {
        data.push(fullSmooth);
        seriesConfigs.push({
          label: `${key} (trend)`,
          scale: scaleKey,
          stroke: colorDull,
          width: 2.5,
          spanGaps: false,
          points: { show: false },
          value: (_u, value) => formatNum(value),
        });
        colArrays.push(fullSmooth);
      }

      scaleArrays[scaleKey] = colArrays;
      scales[scaleKey] = {
        distr: useLogScale ? 3 : 1,
        range: (_u, dataMin, dataMax) => {
          const c = yClipRef.current?.[scaleKey];
          if (c) return [c.min, c.max];
          return [dataMin, dataMax];
        },
      };
      axes.push({
        scale: scaleKey,
        side: ki % 2 === 0 ? 3 : 1,
        stroke: color,
        label: key,
        labelSize: 14,
        grid: { show: ki === 0, stroke: "rgba(255,255,255,0.06)" },
        ticks: { stroke: "rgba(255,255,255,0.15)" },
        size: 60,
        values: (_u, ticks) => ticks.map((tk) => formatNum(tk)),
      });
    }

    let yClip: Record<string, { min: number; max: number }> | null = null;
    if (clipOutliers && xs.length >= 10) {
      yClip = {};
      for (const scaleKey of Object.keys(scaleArrays)) {
        const vals: number[] = [];
        for (const arr of scaleArrays[scaleKey]) {
          for (const v of arr) {
            if (v !== null && Number.isFinite(v)) vals.push(v as number);
          }
        }
        if (vals.length >= 10) {
          vals.sort((a, b) => a - b);
          const lo = vals[Math.floor(vals.length * 0.02)];
          const hi = vals[Math.ceil(vals.length * 0.98) - 1];
          if (Number.isFinite(lo) && Number.isFinite(hi) && lo !== hi) {
            yClip[scaleKey] = { min: lo, max: hi };
          }
        }
      }
    }

    return { data: data as uPlot.AlignedData, seriesConfigs, scales, axes, yClip };
  }, [series, activeKeys, smoothing, plotStride, useLogScale, showTrend, clipOutliers]);

  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);
  useEffect(() => {
    yClipRef.current = built.yClip;
  }, [built.yClip]);

  const isZoomedRef = useRef(false);
  useEffect(() => {
    isZoomedRef.current = isZoomed;
  }, [isZoomed]);

  const hasData = (built.data[0]?.length ?? 0) > 1;
  const structuralKey = useMemo(
    () => `${activeKeys.join("|")}|trend=${showTrend}|log=${useLogScale}|has=${hasData}`,
    [activeKeys, showTrend, useLogScale, hasData],
  );

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

  const totalPoints = built.data[0]?.length ?? 0;

  return (
    <div className="bg-surface rounded-xl border border-border overflow-hidden flex flex-col">
      <div className="px-4 py-3 flex items-center justify-between border-b border-border">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-accent" />
          <h2 className="text-sm font-medium text-text">Loss Graph</h2>
          <span className="text-xs text-text-muted">
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
          className="px-3 py-1 rounded-md text-xs bg-white/5 hover:bg-white/10 text-text-muted hover:text-text border border-border"
        >
          Refresh
        </button>
      </div>

      <div className="px-4 pt-4 pb-4 flex flex-col">
        <div
          className="bg-black/30 rounded-lg border border-border relative select-none"
          style={{ minHeight: 280, height: 320 }}
        >
          {!hasData ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-text-muted">
              {status === "error" ? "Failed to load loss logs." : "Waiting for loss points…"}
            </div>
          ) : (
            <>
              {isZoomed && (
                <button
                  type="button"
                  onClick={handleResetZoom}
                  className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs bg-running/80 hover:bg-running text-bg border border-running/50"
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

      <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-black/20 border border-border rounded-lg p-3">
          <label className="block text-xs text-text-muted mb-2">Display</label>
          <div className="flex flex-wrap gap-2">
            <ToggleButton checked={showTrend} onClick={() => setShowTrend((v) => !v)} label="Trend" />
            <ToggleButton checked={useLogScale} onClick={() => setUseLogScale((v) => !v)} label="Log Y" />
            <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers((v) => !v)} label="Clip outliers" />
          </div>
        </div>

        <div className="bg-black/20 border border-border rounded-lg p-3">
          <label className="block text-xs text-text-muted mb-2">Series</label>
          {lossKeys.length === 0 ? (
            <div className="text-sm text-text-muted">No loss keys found yet.</div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {lossKeys.map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setEnabled((prev) => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                  className={[
                    "px-3 py-1 rounded-md text-xs border transition-colors",
                    enabled[k] === false
                      ? "bg-surface text-text-muted border-border"
                      : "bg-surface text-text border-border hover:bg-white/5",
                  ].join(" ")}
                  aria-pressed={enabled[k] !== false}
                >
                  <span className="inline-block h-2 w-2 rounded-full mr-2" style={{ background: strokeForKey(k) }} />
                  {k}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="bg-black/20 border border-border rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <label className="block text-xs text-text-muted">Smoothing</label>
            <span className="text-xs text-text">{smoothing}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={smoothing}
            onChange={(e) => setSmoothing(Number(e.target.value))}
            className="w-full accent-accent"
          />
        </div>

        <div className="bg-black/20 border border-border rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <label className="block text-xs text-text-muted">Plot stride</label>
            <span className="text-xs text-text">every {plotStride} pt</span>
          </div>
          <input
            type="range"
            min={1}
            max={20}
            value={plotStride}
            onChange={(e) => setPlotStride(Number(e.target.value))}
            className="w-full accent-accent"
          />
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
          background: rgba(212, 160, 84, 0.15);
          border: 1px solid rgba(212, 160, 84, 0.4);
        }
      `}</style>
    </div>
  );
}
