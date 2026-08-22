"use client";

import { inputClassName } from "@/components/ui/Input";

interface ComposedPromptPreviewProps {
  label: string;
  value: string;
}

export default function ComposedPromptPreview({ label, value }: ComposedPromptPreviewProps) {
  return (
    <div>
      <label className="block text-xs font-medium text-text-secondary mb-1">{label}</label>
      <textarea
        readOnly
        value={value}
        rows={12}
        className={`${inputClassName} resize-none font-mono text-xs leading-relaxed min-h-[280px]`}
        placeholder="Add sections to build a prompt…"
      />
    </div>
  );
}
