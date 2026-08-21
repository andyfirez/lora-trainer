"use client";

import { use } from "react";
import { samplingsApi } from "@/lib/api/samplings";
import { useCancelRunnable } from "@/hooks/useCancelRunnable";
import { useRunnableDetail } from "@/hooks/useRunnableDetail";
import { useRunnableActions } from "@/hooks/useRunnableActions";
import SamplingRunPanel from "@/components/sampling/SamplingRunPanel";
import StopJobDialog from "@/components/StopJobDialog";
import RunnableDetailLayout, { RunnableDetailNotFound } from "@/components/runnable/RunnableDetailLayout";
import RunnableEnqueueCancelButtons from "@/components/runnable/RunnableEnqueueCancelButtons";
import YamlViewer from "@/components/YamlViewer";
import type { SamplingResponse } from "@/types";

interface Props {
  params: Promise<{ id: string }>;
}

export default function SamplingDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const id = Number(idParam);
  const { data: sampling, isLoading, mutate } = useRunnableDetail(`/samplings/${id}`, () => samplingsApi.get(id));
  const { handleEnqueue } = useRunnableActions(id, samplingsApi, () => void mutate());
  const cancelSampling = useCancelRunnable<SamplingResponse>(samplingsApi, () => void mutate(), {
    errorMessage: "Failed to stop sampling",
  });

  if (!isLoading && !sampling) {
    return <RunnableDetailNotFound message="Sampling not found" />;
  }

  return (
    <>
      <RunnableDetailLayout
        className="space-y-6 max-w-4xl"
        backHref="/sampling"
        backAriaLabel="Back to sampling"
        backIconSize={18}
        title={sampling?.name ?? ""}
        status={sampling?.status ?? "draft"}
        isLoading={isLoading}
        notFoundMessage="Sampling not found"
        actions={
          sampling ? (
            <RunnableEnqueueCancelButtons
              status={sampling.status}
              onEnqueue={handleEnqueue}
              onCancel={() => cancelSampling.requestCancel(sampling)}
            />
          ) : undefined
        }
      >
        {sampling && (
          <>
            <SamplingRunPanel sampling={sampling} />
            <div className="space-y-2">
              <h2 className="text-sm font-medium text-muted">Config YAML</h2>
              <YamlViewer value={sampling.config_yaml} downloadFilename={`${sampling.name}.yaml`} />
            </div>
          </>
        )}
      </RunnableDetailLayout>

      <StopJobDialog
        open={cancelSampling.dialogRunnable != null}
        jobName={cancelSampling.dialogRunnable?.name ?? ""}
        canSaveCheckpoint={cancelSampling.canSaveCheckpoint}
        loading={cancelSampling.loading}
        error={cancelSampling.error}
        onClose={cancelSampling.closeDialog}
        onStopNow={() =>
          cancelSampling.dialogRunnable &&
          void cancelSampling.executeCancel(cancelSampling.dialogRunnable, false)
        }
        onSaveAndStop={() =>
          cancelSampling.dialogRunnable &&
          void cancelSampling.executeCancel(cancelSampling.dialogRunnable, true)
        }
      />
    </>
  );
}
