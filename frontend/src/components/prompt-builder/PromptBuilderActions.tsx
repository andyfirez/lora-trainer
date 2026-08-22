"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Button from "@/components/ui/Button";
import { applyPromptsToPlayground, savePlaygroundState } from "@/lib/playgroundState";

interface PromptBuilderActionsProps {
  positivePrompt: string;
  negativePrompt: string;
}

export default function PromptBuilderActions({
  positivePrompt,
  negativePrompt,
}: PromptBuilderActionsProps) {
  const router = useRouter();
  const [status, setStatus] = useState<string | null>(null);

  async function copyCombined() {
    const lines = [positivePrompt.trim(), negativePrompt.trim()].filter(Boolean);
    await navigator.clipboard.writeText(lines.join("\n"));
    setStatus("Copied to clipboard");
    setTimeout(() => setStatus(null), 2000);
  }

  function sendToSampling() {
    const next = applyPromptsToPlayground(positivePrompt, negativePrompt);
    savePlaygroundState(next);
    router.push("/sampling");
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button variant="secondary" size="sm" onClick={() => void copyCombined()}>
        Copy prompts
      </Button>
      <Button variant="sampling" size="sm" onClick={sendToSampling}>
        Send to Sampling
      </Button>
      {status && <span className="text-xs text-muted">{status}</span>}
    </div>
  );
}
