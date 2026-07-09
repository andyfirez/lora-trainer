"use client";

import { useState } from "react";
import { Modal, ModalFooter, Button } from "@/components/ui";
import { inputClass, labelClass } from "@/components/ui/Input";
import type { TaggingMode } from "@/types";

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (options: {
    mode: TaggingMode;
    threshold: number;
    model: string;
    strip_rating: boolean;
  }) => Promise<void>;
}

const MODES: { value: TaggingMode; label: string; description: string }[] = [
  { value: "if_empty", label: "If empty", description: "Only tag images without captions" },
  { value: "overwrite", label: "Overwrite", description: "Replace existing captions" },
  { value: "append", label: "Append", description: "Add predicted tags to existing ones" },
];

export default function AutoTagModal({ open, onClose, onSubmit }: Props) {
  const [mode, setMode] = useState<TaggingMode>("if_empty");
  const [threshold, setThreshold] = useState(0.35);
  const [model, setModel] = useState("wd-v1-4-convnextv2-tagger-v2");
  const [stripRating, setStripRating] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit({ mode, threshold, model, strip_rating: stripRating });
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start auto-tagging");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Auto-tag with WD14" size="md">
      <p className="text-sm text-text-muted mb-4">
        Runs in the background via the job queue. Uses SmilingWolf booru tagger (ONNX).
      </p>
      {error && <div className="text-error text-sm mb-4">{error}</div>}
      <form onSubmit={(event) => void handleSubmit(event)} className="space-y-4">
        <div>
          <label className={labelClass}>Mode</label>
          <div className="space-y-2">
            {MODES.map((item) => (
              <label key={item.value} className="flex items-start gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value={item.value}
                  checked={mode === item.value}
                  onChange={() => setMode(item.value)}
                  className="mt-1"
                />
                <span>
                  <span className="text-sm text-text block">{item.label}</span>
                  <span className="text-xs text-text-muted">{item.description}</span>
                </span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className={labelClass}>
            Confidence threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.01}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className={labelClass}>Model</label>
          <select
            value={model}
            onChange={(event) => setModel(event.target.value)}
            className={inputClass}
          >
            <option value="wd-v1-4-convnextv2-tagger-v2">WD v1.4 ConvNeXt v2</option>
            <option value="wd-v1-4-vit-tagger-v2">WD v1.4 ViT v2</option>
            <option value="wd-v1-4-swinv2-tagger-v2">WD v1.4 Swin v2</option>
            <option value="wd-v1-4-moat-tagger-v2">WD v1.4 MOAT v2</option>
          </select>
        </div>

        <label className="flex items-center gap-2 text-sm text-text cursor-pointer">
          <input
            type="checkbox"
            checked={stripRating}
            onChange={(event) => setStripRating(event.target.checked)}
          />
          Strip rating tags (rating:*)
        </label>

        <ModalFooter className="justify-stretch">
          <Button type="button" variant="secondary" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button type="submit" disabled={submitting} className="flex-1">
            {submitting ? "Starting…" : "Start auto-tagging"}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
