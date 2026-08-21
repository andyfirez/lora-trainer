"use client";

import { strokeForKey } from "./lossGraphMath";

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "px-3 py-1 rounded-md text-xs border transition-colors",
        checked
          ? "bg-accent-muted text-accent border-accent/30 hover:bg-accent/20"
          : "bg-surface text-muted border-border hover:bg-white/5",
      ].join(" ")}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}

export interface LossGraphControlsProps {
  lossKeys: string[];
  showTrend: boolean;
  useLogScale: boolean;
  clipOutliers: boolean;
  smoothing: number;
  plotStride: number;
  enabled: Record<string, boolean>;
  onToggleTrend: () => void;
  onToggleLogScale: () => void;
  onToggleClipOutliers: () => void;
  onSmoothingChange: (value: number) => void;
  onPlotStrideChange: (value: number) => void;
  onToggleSeries: (key: string) => void;
}

export default function LossGraphControls({
  lossKeys,
  showTrend,
  useLogScale,
  clipOutliers,
  smoothing,
  plotStride,
  enabled,
  onToggleTrend,
  onToggleLogScale,
  onToggleClipOutliers,
  onSmoothingChange,
  onPlotStrideChange,
  onToggleSeries,
}: LossGraphControlsProps) {
  return (
    <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
      <div className="bg-black/20 border border-border rounded-lg p-3">
        <label className="block text-xs text-muted mb-2">Display</label>
        <div className="flex flex-wrap gap-2">
          <ToggleButton checked={showTrend} onClick={onToggleTrend} label="Trend" />
          <ToggleButton checked={useLogScale} onClick={onToggleLogScale} label="Log Y" />
          <ToggleButton checked={clipOutliers} onClick={onToggleClipOutliers} label="Clip outliers" />
        </div>
      </div>

      <div className="bg-black/20 border border-border rounded-lg p-3">
        <label className="block text-xs text-muted mb-2">Series</label>
        {lossKeys.length === 0 ? (
          <div className="text-sm text-muted">No loss keys found yet.</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {lossKeys.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => onToggleSeries(k)}
                className={[
                  "px-3 py-1 rounded-md text-xs border transition-colors",
                  enabled[k] === false
                    ? "bg-surface text-muted border-border"
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
          <label className="block text-xs text-muted">Smoothing</label>
          <span className="text-xs text-text">{smoothing}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={smoothing}
          onChange={(e) => onSmoothingChange(Number(e.target.value))}
          className="w-full accent-accent"
        />
      </div>

      <div className="bg-black/20 border border-border rounded-lg p-3">
        <div className="flex items-center justify-between mb-1">
          <label className="block text-xs text-muted">Plot stride</label>
          <span className="text-xs text-text">every {plotStride} pt</span>
        </div>
        <input
          type="range"
          min={1}
          max={20}
          value={plotStride}
          onChange={(e) => onPlotStrideChange(Number(e.target.value))}
          className="w-full accent-accent"
        />
      </div>
    </div>
  );
}
