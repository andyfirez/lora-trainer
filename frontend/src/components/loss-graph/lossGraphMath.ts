import type { LossPoint } from "@/hooks/useLossLog";
import type { MutableRefObject } from "react";
import type uPlot from "uplot";

export const FALLBACK_CANVAS_HEIGHT = 360;
export const MIN_CANVAS_HEIGHT = 160;

const PALETTE = [
  "rgba(217, 119, 6, 1)",
  "rgba(245, 158, 11, 1)",
  "rgba(234, 88, 12, 1)",
  "rgba(180, 83, 9, 1)",
  "rgba(251, 191, 36, 1)",
  "rgba(196, 120, 70, 1)",
  "rgba(239, 68, 68, 1)",
  "rgba(212, 165, 116, 1)",
];

export function formatNum(v: number) {
  if (!Number.isFinite(v)) return "";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(3);
  if (abs >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

export function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
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

export function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
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

export function strokeForKey(key: string) {
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

export interface BuiltChartData {
  data: uPlot.AlignedData;
  seriesConfigs: uPlot.Series[];
  scales: uPlot.Scales;
  axes: uPlot.Axis[];
  yClip: Record<string, { min: number; max: number }> | null;
}

export interface BuildChartDataOptions {
  series: Record<string, LossPoint[]>;
  activeKeys: string[];
  smoothing: number;
  plotStride: number;
  useLogScale: boolean;
  showTrend: boolean;
  clipOutliers: boolean;
  yClipRef: MutableRefObject<Record<string, { min: number; max: number }> | null>;
}

export function buildChartData({
  series,
  activeKeys,
  smoothing,
  plotStride,
  useLogScale,
  showTrend,
  clipOutliers,
  yClipRef,
}: BuildChartDataOptions): BuiltChartData {
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
}
