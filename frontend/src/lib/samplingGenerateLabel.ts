export type GeneratePrimaryLabel = "Generate" | "Interrupt";

export function resolveGenerateLabel(input: {
  optimistic: boolean;
  sessionQueued: boolean;
  sessionRunning: boolean;
}): GeneratePrimaryLabel {
  if (input.optimistic || input.sessionQueued || input.sessionRunning) return "Interrupt";
  return "Generate";
}

export function resolveGenerateBusy(input: {
  label: GeneratePrimaryLabel;
  submitting: boolean;
}): boolean {
  if (input.label === "Interrupt") return false;
  return input.submitting;
}
