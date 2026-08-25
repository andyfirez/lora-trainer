"use client";

import ExperimentRunPanel from "@/components/experiments/ExperimentRunPanel";
import type { SamplingResponse } from "@/types";

interface SamplingRunPanelProps {
  sampling: SamplingResponse;
}

export default function SamplingRunPanel({ sampling }: SamplingRunPanelProps) {
  return <ExperimentRunPanel experiment={sampling} />;
}
