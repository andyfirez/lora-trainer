"use client";

import { useState } from "react";
import ExperimentField from "@/components/experiments/ExperimentField";
import AddLoraModal from "@/components/playground/AddLoraModal";
import Button from "@/components/ui/Button";
import { formSectionTitleClass } from "@/components/ui/FormSection";
import PromptTokenCount from "@/components/sampling/PromptTokenCount";
import { appendLoraTag } from "@/lib/loraPrompt";
import { getParameters, setParameter } from "@/lib/experimentUtils";

interface PlaygroundPromptBarProps {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}

export default function PlaygroundPromptBar({ config, onChange }: PlaygroundPromptBarProps) {
  const parameters = getParameters(config);
  const [promptOpen, setPromptOpen] = useState(true);
  const [loraModalOpen, setLoraModalOpen] = useState(false);

  const promptParam = parameters.prompt ?? { mode: "fixed", value: "" };
  const negativeParam = parameters.negative_prompt ?? { mode: "fixed", value: "" };
  const currentPrompt = String(
    promptParam.mode === "vary" ? (promptParam.values?.[0] ?? "") : (promptParam.value ?? ""),
  );
  const currentNegative = String(
    negativeParam.mode === "vary" ? (negativeParam.values?.[0] ?? "") : (negativeParam.value ?? ""),
  );
  const baseModelName = String(
    parameters.base_model_name?.mode === "vary"
      ? (parameters.base_model_name.values?.[0] ?? "")
      : (parameters.base_model_name?.value ?? ""),
  );

  function updatePrompt(nextPrompt: string) {
    onChange(setParameter(config, "prompt", { mode: "fixed", value: nextPrompt }));
  }

  return (
    <section className="border-b border-border p-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setPromptOpen(!promptOpen)}
          className={`${formSectionTitleClass} mb-0 flex-1 text-left flex items-center justify-between`}
        >
          Prompt
          <span className="text-muted text-xs">{promptOpen ? "▲" : "▼"}</span>
        </button>
        <PromptTokenCount text={currentPrompt} baseModelName={baseModelName} />
        <Button type="button" variant="ghost" size="sm" onClick={() => setLoraModalOpen(true)}>
          Add LoRA
        </Button>
      </div>
      {promptOpen && (
        <div className="mt-3 space-y-2">
          <ExperimentField
            label=""
            param={promptParam}
            onChange={(param) => onChange(setParameter(config, "prompt", param))}
            multiline
            placeholder="subject, style, quality tags"
            allowVary={false}
          />
          <ExperimentField
            label="Negative prompt"
            param={negativeParam}
            onChange={(param) => onChange(setParameter(config, "negative_prompt", param))}
            multiline
            placeholder="low quality, blurry"
            allowVary={false}
            accessory={<PromptTokenCount text={currentNegative} baseModelName={baseModelName} />}
          />
        </div>
      )}
      <AddLoraModal
        open={loraModalOpen}
        onClose={() => setLoraModalOpen(false)}
        onConfirm={(name, weight) => updatePrompt(appendLoraTag(currentPrompt, name, weight))}
      />
    </section>
  );
}
