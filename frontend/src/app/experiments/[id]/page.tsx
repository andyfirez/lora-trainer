"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import { Copy, Loader2, RotateCcw, Square } from "lucide-react";
import ExperimentRunPanel from "@/components/experiments/ExperimentRunPanel";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import SamplingConfigForm from "@/components/SamplingConfigForm";
import Alert from "@/components/ui/Alert";
import Button from "@/components/ui/Button";
import RunnableDetailLayout, { RunnableDetailNotFound } from "@/components/runnable/RunnableDetailLayout";
import { useRunnableDetail } from "@/hooks/useRunnableDetail";
import { samplingsApi } from "@/lib/api/samplings";
import { countCombinations, experimentSummary, getParameters, planGrids } from "@/lib/experimentUtils";
import { canCancel, cancelActionLabel } from "@/lib/runnableStatus";
import type { SamplingResponse } from "@/types";

interface Props {
  params: Promise<{ id: string }>;
}

function experimentSubtitle(experiment: SamplingResponse): string {
  const parameters = getParameters(experiment.config);
  const grid = (experiment.config.grid as { x_axis?: string; y_axis?: string }) ?? {};
  const imageCount = countCombinations(parameters);
  const gridPlan = planGrids(parameters, grid);
  const parts = [`${imageCount} image${imageCount === 1 ? "" : "s"}`, experimentSummary(parameters)];
  if (gridPlan.gridCount > 0) {
    parts.push(`${gridPlan.gridCount} grid${gridPlan.gridCount === 1 ? "" : "s"}`);
  }
  return parts.join(" · ");
}

function ExperimentStatusBanner({ experiment }: { experiment: SamplingResponse }) {
  if (experiment.status === "queued" && experiment.queue_position != null) {
    return <Alert variant="info">In queue — position #{experiment.queue_position}</Alert>;
  }
  if (experiment.status === "failed" && experiment.error_message) {
    return (
      <Alert variant="error">
        <strong>Experiment failed:</strong> {experiment.error_message}
      </Alert>
    );
  }
  if (experiment.status === "completed") {
    return <Alert variant="success">Experiment completed successfully.</Alert>;
  }
  return null;
}

function canRepeat(status: SamplingResponse["status"]): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

export default function ExperimentDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const experimentId = Number(idParam);
  const router = useRouter();
  const { data: experiment, isLoading, mutate } = useRunnableDetail(
    `/samplings/${experimentId}`,
    () => samplingsApi.get(experimentId),
  );
  const [repeating, setRepeating] = useState(false);

  if (!isLoading && (!experiment || experiment.kind !== "experiment")) {
    return <RunnableDetailNotFound message="Experiment not found" />;
  }

  const handleRepeat = async () => {
    setRepeating(true);
    try {
      await samplingsApi.enqueue(experimentId);
      await mutate();
    } finally {
      setRepeating(false);
    }
  };

  const handleCancel = async () => {
    await samplingsApi.cancel(experimentId);
    await mutate();
  };

  const handleCopy = () => {
    router.push(`/experiments/new?copyFrom=${experimentId}`);
  };

  return (
    <RunnableDetailLayout
      backHref="/experiments"
      backAriaLabel="Back to experiments"
      title={experiment?.name ?? ""}
      subtitle={experiment ? experimentSubtitle(experiment) : undefined}
      status={experiment?.status ?? "draft"}
      isLoading={isLoading}
      notFoundMessage="Experiment not found"
      banner={experiment ? <ExperimentStatusBanner experiment={experiment} /> : undefined}
      actions={
        experiment ? (
          <>
            {canRepeat(experiment.status) && (
              <Button variant="secondary" size="sm" onClick={() => void handleRepeat()} disabled={repeating}>
                {repeating ? <Loader2 className="animate-spin" size={13} /> : <RotateCcw size={13} />}
                Repeat
              </Button>
            )}
            {canRepeat(experiment.status) && (
              <Button variant="secondary" size="sm" onClick={handleCopy}>
                <Copy size={13} /> Copy
              </Button>
            )}
            {canCancel(experiment.status) && (
              <Button variant="danger" size="sm" onClick={() => void handleCancel()}>
                <Square size={13} /> {cancelActionLabel(experiment.status)}
              </Button>
            )}
          </>
        ) : undefined
      }
    >
      {experiment && (
        <div className="mx-auto max-w-4xl space-y-6">
          <ExperimentRunPanel experiment={experiment} hideLogs />
          {(experiment.status === "running" ||
            experiment.status === "queued" ||
            experiment.status === "completed" ||
            experiment.status === "failed" ||
            experiment.status === "cancelled") && (
            <LiveLogsPanel
              swrKey={`/samplings/${experiment.id}/logs`}
              fetcher={() => samplingsApi.getLogs(experiment.id, 500)}
              isRunning={experiment.status === "running"}
              showLogs
              status={experiment.status}
              title="Experiment Logs"
            />
          )}
          <div className="rounded-xl border border-border bg-surface p-4">
            <h2 className="text-sm font-medium text-muted mb-3">Configuration</h2>
            <SamplingConfigForm config={experiment.config} onChange={() => {}} readOnly />
          </div>
        </div>
      )}
    </RunnableDetailLayout>
  );
}
