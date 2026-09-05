"use client";

import { usePromptTokenCount } from "@/hooks/usePromptTokenCount";

interface PromptTokenCountProps {
  text: string;
  baseModelName: string;
}

export default function PromptTokenCount({ text, baseModelName }: PromptTokenCountProps) {
  const counts = usePromptTokenCount(text, baseModelName);
  if (!counts) return null;
  const over = counts.tokens > counts.window;
  return (
    <span className={`text-xs tabular-nums ${over ? "text-warning" : "text-muted"}`}>
      {counts.tokens}/{counts.window}
      {counts.chunks > 1 ? ` · ${counts.chunks} chunks` : ""}
    </span>
  );
}
