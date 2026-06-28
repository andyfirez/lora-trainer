"use client";

import { useState } from "react";
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

  if (!open) return null;

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
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-6 w-full max-w-md space-y-4">
        <h2 className="text-lg font-semibold text-white">Auto-tag with WD14</h2>
        <p className="text-sm text-[var(--muted)]">
          Runs in the background via the job queue. Uses SmilingWolf booru tagger (ONNX).
        </p>
        {error && <div className="text-red-400 text-sm">{error}</div>}
        <form onSubmit={(event) => void handleSubmit(event)} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--muted)] mb-2">Mode</label>
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
                    <span className="text-sm text-white block">{item.label}</span>
                    <span className="text-xs text-[var(--muted)]">{item.description}</span>
                  </span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">
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
            <label className="block text-xs text-[var(--muted)] mb-1">Model</label>
            <select
              value={model}
              onChange={(event) => setModel(event.target.value)}
              className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm text-white focus:outline-none focus:border-[var(--accent)]"
            >
              <option value="wd-v1-4-convnextv2-tagger-v2">WD v1.4 ConvNeXt v2</option>
              <option value="wd-v1-4-vit-tagger-v2">WD v1.4 ViT v2</option>
              <option value="wd-v1-4-swinv2-tagger-v2">WD v1.4 Swin v2</option>
              <option value="wd-v1-4-moat-tagger-v2">WD v1.4 MOAT v2</option>
            </select>
          </div>

          <label className="flex items-center gap-2 text-sm text-white cursor-pointer">
            <input
              type="checkbox"
              checked={stripRating}
              onChange={(event) => setStripRating(event.target.checked)}
            />
            Strip rating tags (rating:*)
          </label>

          <div className="flex gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 border border-[var(--border)] rounded-lg py-2 text-[var(--muted)] hover:text-white text-sm"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg py-2 text-sm font-medium disabled:opacity-50"
            >
              {submitting ? "Starting…" : "Start auto-tagging"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
