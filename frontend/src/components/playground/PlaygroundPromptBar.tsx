"use client";

import ExperimentField from "@/components/experiments/ExperimentField";
import { getParameters, setParameter } from "@/lib/experimentUtils";

interface PlaygroundPromptBarProps {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}

export default function PlaygroundPromptBar({ config, onChange }: PlaygroundPromptBarProps) {
  const parameters = getParameters(config);

  return (
    <div className="space-y-2 px-3 py-2">
      <ExperimentField
        label="Prompt"
        param={parameters.prompt ?? { mode: "fixed", value: "" }}
        onChange={(param) => onChange(setParameter(config, "prompt", param))}
        multiline
        placeholder="subject, style, quality tags"
        allowVary={false}
      />
      <ExperimentField
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
