"use client";

import SweepField from "@/components/sweep/SweepField";
import { getParameters, setParameter } from "@/lib/sweepUtils";

interface PlaygroundPromptBarProps {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}

export default function PlaygroundPromptBar({ config, onChange }: PlaygroundPromptBarProps) {
  const parameters = getParameters(config);

  return (
    <div className="space-y-2 px-3 py-2">
      <SweepField
        label="Prompt"
        param={parameters.prompt ?? { mode: "fixed", value: "" }}
        onChange={(param) => onChange(setParameter(config, "prompt", param))}
        multiline
        placeholder="subject, style, quality tags"
        allowVary={false}
      />
      <SweepField
        label="Negative prompt"
        param={parameters.negative_prompt ?? { mode: "fixed", value: "" }}
        onChange={(param) => onChange(setParameter(config, "negative_prompt", param))}
        multiline
        placeholder="low quality, blurry"
        allowVary={false}
      />
    </div>
  );
}
