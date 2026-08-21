"use client";

import Link from "next/link";
import { Plus, X } from "lucide-react";
import FieldHint from "@/components/FieldHint";
import FormSection from "@/components/ui/FormSection";
import { inputClassName } from "@/components/ui/Input";
import { trainHint } from "@/lib/trainParameterMetadata";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainField, TrainNumberInput, TrainSelectInput } from "@/components/train/TrainFormFields";

interface TrainConceptsSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainConceptsSection({ form }: TrainConceptsSectionProps) {
  const {
    config,
    set,
    concepts,
    datasets,
    datasetsLoading,
    datasetById,
    isDatasetCompatible,
    datasetOptions,
    trainResolution,
    trainEnableBucket,
    updateConcept,
    addConcept,
    removeConcept,
  } = form;

  return (
    <FormSection title="Data">
      <TrainNumberInput
        label="Resolution"
        value={config.resolution as number | null | undefined}
        onChange={(value) => set("resolution", value)}
        min={64}
        max={2048}
        step={64}
        placeholder="1024"
        paramKey="resolution"
      />
      <label className="flex items-center gap-2 text-sm text-text mt-2 cursor-pointer">
        <input
          type="checkbox"
          checked={trainEnableBucket}
          onChange={(e) => set("enable_bucket", e.target.checked)}
          className="rounded"
        />
        <span className="flex items-center">
          Enable aspect-ratio bucketing
          {trainHint("enable_bucket").hint && (
            <FieldHint hint={trainHint("enable_bucket").hint!} hintAnchor="enable_bucket" />
          )}
        </span>
      </label>
      <div className="space-y-3 mt-2">
        <div className="text-xs font-medium text-muted">Concepts</div>
        {datasetsLoading ? (
          <div className="text-sm text-muted">Loading datasets…</div>
        ) : !datasets?.length ? (
          <div className="rounded-lg border border-dashed border-border p-6 text-center space-y-3">
            <p className="text-sm text-muted">No datasets yet. Create a dataset to specify training data.</p>
            <Link
              href="/datasets"
              className="inline-flex items-center gap-1.5 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium"
            >
              Create Dataset
            </Link>
          </div>
        ) : (
          <>
            {concepts.map((concept, index) => {
              const selectedDataset = datasetById(concept.dataset_id as number | undefined);
              return (
                <div key={index} className="relative rounded-lg border border-border p-4 bg-bg">
                  <button
                    type="button"
                    onClick={() => removeConcept(index)}
                    className="absolute top-2 right-2 p-1 rounded hover:bg-white/10 text-muted hover:text-error"
                  >
                    <X size={13} />
                  </button>
                  <div className="text-xs text-muted mb-3">Concept {index + 1}</div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div className="md:col-span-1">
                      <TrainSelectInput
                        label="Dataset"
                        value={concept.dataset_id != null ? String(concept.dataset_id) : ""}
                        onChange={(value) => updateConcept(index, "dataset_id", Number(value))}
                        options={[{ value: "", label: "Select dataset…" }, ...datasetOptions]}
                        paramKey="concepts.dataset_id"
                      />
                      {selectedDataset && (
                        <p className="text-xs text-muted mt-1 break-all">{selectedDataset.relative_path}</p>
                      )}
                      {selectedDataset && !isDatasetCompatible(selectedDataset) && (
                        <p className="text-xs text-warning mt-1">
                          Dataset must be prepared at {trainResolution}px. Open the dataset page to crop and bake
                          images.
                        </p>
                      )}
                      {concept.dataset_id == null && <p className="text-xs text-error mt-1">Select a dataset</p>}
                      {concept.dataset_id != null && !selectedDataset && (
                        <p className="text-xs text-error mt-1">Dataset not found</p>
                      )}
                    </div>
                    <TrainField label="Trigger Words" {...trainHint("concepts.trigger_words")}>
                      <input
                        type="text"
                        className={inputClassName}
                        value={((concept.trigger_words as string[] | undefined) ?? []).join(", ")}
                        onChange={(e) =>
                          updateConcept(
                            index,
                            "trigger_words",
                            e.target.value
                              .split(",")
                              .map((word) => word.trim())
                              .filter(Boolean),
                          )
                        }
                        placeholder="ohwx, person"
                      />
                    </TrainField>
                    <TrainField label="Caption Extension" {...trainHint("concepts.caption_extension")}>
                      <input
                        type="text"
                        className={inputClassName}
                        value={(concept.caption_extension as string) ?? ".txt"}
                        onChange={(e) => updateConcept(index, "caption_extension", e.target.value)}
                        placeholder=".txt"
                      />
                    </TrainField>
                    <TrainField label="Repeats" {...trainHint("concepts.repeats")}>
                      <input
                        type="number"
                        className={inputClassName}
                        value={(concept.repeats as number) ?? 1}
                        min={1}
                        onChange={(e) => updateConcept(index, "repeats", Number(e.target.value))}
                      />
                    </TrainField>
                  </div>
                </div>
              );
            })}
            <button
              type="button"
              onClick={addConcept}
              className="flex items-center gap-1.5 text-sm text-muted hover:text-text border border-dashed border-border hover:border-text/30 rounded-lg px-3 py-2 w-full justify-center transition-colors"
            >
              <Plus size={13} /> Add Concept
            </button>
          </>
        )}
      </div>
    </FormSection>
  );
}
