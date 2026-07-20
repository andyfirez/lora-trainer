"use client";

import { selectClassName } from "@/components/ui/Select";
import {
  SWEEP_PARAM_LABELS,
  type SweepParamKey,
  getParameters,
  planGrids,
  setGridLayout,
  varyKeysWithValues,
} from "@/lib/sweepUtils";

type Config = Record<string, unknown>;

interface GridLayoutSectionProps {
  config: Config;
  onChange: (config: Config) => void;
}

const sectionClass = "bg-surface rounded-xl border border-border p-5 space-y-4";
const sectionTitleClass = "text-sm font-semibold text-text mb-3 font-display";

export default function GridLayoutSection({ config, onChange }: GridLayoutSectionProps) {
  const parameters = getParameters(config);
  const varyKeys = varyKeysWithValues(parameters);
  const grid = (config.grid as { x_axis?: string | null; y_axis?: string | null }) ?? {};
  const preview = planGrids(parameters, grid);

  if (varyKeys.length < 2) {
    return null;
  }

  function updateAxis(key: "x_axis" | "y_axis", value: string) {
    onChange(setGridLayout(config, { ...grid, [key]: value || null }));
  }

  return (
    <section className={sectionClass}>
      <div className={sectionTitleClass}>Grid Layout</div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-muted mb-1">X axis</label>
          <select
            className={selectClassName}
            value={grid.x_axis ?? varyKeys[0]}
            onChange={(e) => updateAxis("x_axis", e.target.value)}
          >
            {varyKeys.map((k) => (
              <option key={k} value={k}>
                {SWEEP_PARAM_LABELS[k as SweepParamKey]}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-muted mb-1">Y axis</label>
          <select
            className={selectClassName}
            value={grid.y_axis ?? varyKeys[1]}
            onChange={(e) => updateAxis("y_axis", e.target.value)}
          >
            {varyKeys.map((k) => (
              <option key={k} value={k}>
                {SWEEP_PARAM_LABELS[k as SweepParamKey]}
              </option>
            ))}
          </select>
        </div>
      </div>
      {preview.sliceKeys.length > 0 && (
        <p className="text-sm text-muted">
          Slice by: {preview.sliceKeys.join(", ")} → {preview.gridCount} grid
          {preview.gridCount !== 1 ? "s" : ""} ({preview.cols}×{preview.rows} each)
        </p>
      )}
      {preview.sliceKeys.length === 0 && (
        <p className="text-sm text-muted">
          1 grid ({preview.cols}×{preview.rows})
        </p>
      )}
    </section>
  );
}
