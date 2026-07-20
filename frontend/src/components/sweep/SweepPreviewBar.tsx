"use client";

import { countCombinations, getParameters, planGrids } from "@/lib/sweepUtils";

type Config = Record<string, unknown>;

interface SweepPreviewBarProps {
  config: Config;
}

export default function SweepPreviewBar({ config }: SweepPreviewBarProps) {
  const parameters = getParameters(config);
  const total = countCombinations(parameters);
  const grid = (config.grid as { x_axis?: string | null; y_axis?: string | null }) ?? {};
  const plan = planGrids(parameters, grid);

  return (
    <div className="sticky bottom-0 z-10 -mx-4 px-4 py-3 bg-bg/95 border-t border-border backdrop-blur-sm">
      <div className="max-w-4xl mx-auto flex items-center justify-between gap-4 text-sm">
        <span className="text-muted">
          {total} image{total !== 1 ? "s" : ""}
          {plan.gridCount > 0 && varyKeysEnough(parameters)
            ? ` · ${plan.gridCount} grid${plan.gridCount !== 1 ? "s" : ""} (${plan.cols}×${plan.rows})`
            : ""}
        </span>
        <span className="text-xs text-muted">Save config, then run from config detail or training job</span>
      </div>
    </div>
  );
}

function varyKeysEnough(parameters: ReturnType<typeof getParameters>): boolean {
  return Object.entries(parameters).some(
    ([, p]) => p?.mode === "vary" && (p.values?.length ?? 0) > 1,
  );
}
