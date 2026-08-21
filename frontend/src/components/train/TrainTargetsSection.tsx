"use client";

import FieldHint from "@/components/FieldHint";
import FormSection from "@/components/ui/FormSection";
import { trainHint } from "@/lib/trainParameterMetadata";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import type { TrainConfig } from "@/lib/trainConfigSanitize";
import { TrainNumberInput, weightDtypeOptions } from "@/components/train/TrainFormFields";

interface TrainTargetsSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainTargetsSection({ form }: TrainTargetsSectionProps) {
  const { config, set, setNested } = form;

  return (
    <FormSection title="Training Targets">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-muted">
              <th className="pb-2 font-medium">Component</th>
              <th className="pb-2 font-medium">Train</th>
              <th className="pb-2 font-medium">Weight Dtype</th>
              <th className="pb-2 font-medium">Learning Rate</th>
            </tr>
          </thead>
          <tbody className="space-y-2">
            {(["unet", "text_encoder_1", "text_encoder_2"] as const).map((part) => {
              const partConfig = (config[part] ?? {}) as TrainConfig;
              const trainHints = trainHint(`${part}.train`);
              const dtypeHints = trainHint(`${part}.weight_dtype`);
              const lrHints = trainHint(`${part}.learning_rate`);
              const isTraining = Boolean(partConfig.train ?? part === "unet");
              return (
                <tr key={part} className="border-t border-border">
                  <td className="py-2 pr-4 text-text font-mono text-xs">{part}</td>
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-1">
                      <input
                        type="checkbox"
                        className="w-4 h-4 accent-accent"
                        checked={isTraining}
                        onChange={(e) => setNested(part, "train", e.target.checked)}
                      />
                      {trainHints.hint && (
                        <FieldHint hint={trainHints.hint} hintAnchor={trainHints.hintAnchor} />
                      )}
                    </div>
                  </td>
                  <td className="py-2">
                    <div className="flex items-center gap-1">
                      <select
                        className="rounded-lg bg-bg border border-border px-2 py-1 text-xs text-text focus:outline-none focus:border-accent"
                        value={(partConfig.weight_dtype as string) ?? "float16"}
                        onChange={(e) => setNested(part, "weight_dtype", e.target.value)}
                      >
                        {weightDtypeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      {dtypeHints.hint && (
                        <FieldHint hint={dtypeHints.hint} hintAnchor={dtypeHints.hintAnchor} />
                      )}
                    </div>
                  </td>
                  <td className="py-2">
                    {isTraining ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          className="rounded-lg bg-bg border border-border px-2 py-1 text-xs text-text focus:outline-none focus:border-accent w-28"
                          value={(partConfig.learning_rate as number) ?? 0.00005}
                          min={0}
                          step={0.00001}
                          placeholder="0.00005"
                          onChange={(e) => {
                            const raw = e.target.value;
                            setNested(part, "learning_rate", raw === "" ? 0.00005 : Number(raw));
                          }}
                        />
                        {lrHints.hint && (
                          <FieldHint hint={lrHints.hint} hintAnchor={lrHints.hintAnchor} />
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t border-border">
        <TrainNumberInput
          label="CLIP Skip"
          value={(config.clip_skip as number | null | undefined) ?? 2}
          onChange={(value) => set("clip_skip", value ?? 2)}
          min={1}
          step={1}
          placeholder="2"
          paramKey="clip_skip"
        />
      </div>
      <p className="text-xs text-muted">
        CLIP hidden layer used for text encoding during training and sampling. Default 2 matches Kohya.
      </p>
    </FormSection>
  );
}
