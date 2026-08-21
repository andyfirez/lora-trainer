"use client";

import { use } from "react";
import { samplingsApi } from "@/lib/api/samplings";
import { useRunnableDetail } from "@/hooks/useRunnableDetail";
import { useRunnableActions } from "@/hooks/useRunnableActions";
import SamplingRunPanel from "@/components/sampling/SamplingRunPanel";
import RunnableDetailLayout, { RunnableDetailNotFound } from "@/components/runnable/RunnableDetailLayout";
import RunnableEnqueueCancelButtons from "@/components/runnable/RunnableEnqueueCancelButtons";
import YamlViewer from "@/components/YamlViewer";

interface Props {
  params: Promise<{ id: string }>;
}

export default function SamplingDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const id = Number(idParam);
  const { data: sampling, isLoading, mutate } = useRunnableDetail(`/samplings/${id}`, () => samplingsApi.get(id));
  const { handleEnqueue, handleCancel } = useRunnableActions(id, samplingsApi, () => void mutate());

  if (!isLoading && !sampling) {
    return <RunnableDetailNotFound message="Sampling not found" />;
  }

  return (
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
            onCancel={handleCancel}
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
  );
}
