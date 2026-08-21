"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, Loader2, Play, RotateCcw, Square } from "lucide-react";
import { lorasApi } from "@/lib/api/loras";
import { useCancelLora } from "@/hooks/useCancelLora";
import { useRunnableDetail } from "@/hooks/useRunnableDetail";
import { canCancel, canEnqueue, cancelActionLabel } from "@/lib/runnableStatus";
import StopJobDialog from "@/components/StopJobDialog";
import LoraRunPanel from "@/components/lora/LoraRunPanel";
import LoraOverviewPanel from "@/components/lora/LoraOverviewPanel";
import LoraCheckpointPanel from "@/components/lora/LoraCheckpointPanel";
import LoraSamplesPanel from "@/components/lora/LoraSamplesPanel";
import LoraConfigPanel from "@/components/lora/LoraConfigPanel";
import Button from "@/components/ui/Button";
import Alert from "@/components/ui/Alert";
import RunnableDetailLayout, { RunnableDetailNotFound } from "@/components/runnable/RunnableDetailLayout";
import { ModalError, ModalFooter } from "@/components/ui/Modal";
import Modal from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Checkbox from "@/components/ui/Checkbox";
import type { LoraResponse } from "@/types";

interface Props {
  params: Promise<{ id: string }>;
}

function LoraStatusBanner({ lora }: { lora: LoraResponse }) {
  if (lora.status === "queued" && lora.queue_position != null) {
    return <Alert variant="info">In queue — position #{lora.queue_position}</Alert>;
  }

  if (lora.status === "running" && lora.save_checkpoint_requested) {
    return (
      <Alert variant="warning" className="flex items-center gap-2 text-warning bg-warning/10">
        <AlertTriangle size={16} />
        Saving checkpoint before stopping…
      </Alert>
    );
  }

  if (lora.status === "failed" && lora.error_message) {
    return (
      <Alert variant="error">
        <strong>Training failed:</strong> {lora.error_message}
      </Alert>
    );
  }

  if (lora.path_missing && lora.status === "completed") {
    return (
      <Alert variant="warning" className="flex items-center gap-2">
        <AlertTriangle size={16} className="text-warning shrink-0" />
        Artifacts not found on disk — paths may be stale.
      </Alert>
    );
  }

  if (lora.status === "completed") {
    return <Alert variant="success">Training completed successfully.</Alert>;
  }

  return null;
}

export default function LoraDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const loraId = Number(idParam);
  const router = useRouter();
  const { data: lora, isLoading, mutate } = useRunnableDetail(`/loras/${loraId}`, () => lorasApi.get(loraId));

  const [showReproduceModal, setShowReproduceModal] = useState(false);
  const [reproduceName, setReproduceName] = useState("");
  const [reproduceEnqueue, setReproduceEnqueue] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [reproduceError, setReproduceError] = useState<string | null>(null);

  const cancelLora = useCancelLora(() => mutate());

  const openReproduceModal = () => {
    setReproduceName(lora ? `${lora.name}-copy` : "");
    setReproduceEnqueue(true);
    setReproduceError(null);
    setShowReproduceModal(true);
  };

  const handleReproduce = async () => {
    if (!reproduceName.trim()) {
      setReproduceError("Name is required");
      return;
    }
    setSubmitting(true);
    setReproduceError(null);
    try {
      const created = await lorasApi.reproduce(loraId, { name: reproduceName.trim(), enqueue: reproduceEnqueue });
      setShowReproduceModal(false);
      router.push(`/loras/${created.id}`);
    } catch (err: unknown) {
      setReproduceError(err instanceof Error ? err.message : "Failed to reproduce training");
    } finally {
      setSubmitting(false);
    }
  };

  const handleEnqueue = async () => {
    await lorasApi.enqueue(loraId);
    mutate();
  };

  const handleResume = async () => {
    await lorasApi.resume(loraId);
    mutate();
  };

  if (!isLoading && !lora) {
    return <RunnableDetailNotFound message="LoRA not found" />;
  }

  const subtitle = lora ? [lora.base_model_name, lora.relative_path].filter(Boolean).join(" · ") : undefined;

  return (
    <>
      <RunnableDetailLayout
        backHref="/loras"
        backAriaLabel="Back to LoRAs"
        title={lora?.name ?? ""}
        subtitle={subtitle}
        status={lora?.status ?? "draft"}
        isLoading={isLoading}
        notFoundMessage="LoRA not found"
        banner={lora ? <LoraStatusBanner lora={lora} /> : undefined}
        actions={
          lora ? (
            <>
              {canEnqueue(lora.status) && (
                <Button variant="success" size="sm" onClick={() => void handleEnqueue()}>
                  <Play size={13} /> Enqueue
                </Button>
              )}
              {lora.can_resume && (
                <Button variant="secondary" size="sm" onClick={() => void handleResume()}>
                  <RotateCcw size={13} /> Resume
                </Button>
              )}
              {canCancel(lora.status) && (
                <Button variant="danger" size="sm" onClick={() => cancelLora.requestCancel(lora)}>
                  <Square size={13} /> {cancelActionLabel(lora.status)}
                </Button>
              )}
              {lora.config_yaml && (
                <Button variant="secondary" size="sm" onClick={openReproduceModal}>
                  <Play size={13} /> Reproduce
                </Button>
              )}
            </>
          ) : undefined
        }
      >
        {lora && (
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <LoraRunPanel lora={lora} lossGraphRunKey={lora.id} />
              <LoraSamplesPanel loraId={loraId} status={lora.status} />
              <LoraConfigPanel name={lora.name} configYaml={lora.config_yaml} />
            </div>
            <div className="space-y-6">
              <LoraOverviewPanel lora={lora} />
              <LoraCheckpointPanel lora={lora} />
            </div>
          </div>
        )}
      </RunnableDetailLayout>

      <StopJobDialog
        open={cancelLora.dialogLora != null}
        jobName={cancelLora.dialogLora?.name ?? ""}
        canSaveCheckpoint={cancelLora.canSaveCheckpoint}
        loading={cancelLora.loading}
        error={cancelLora.error}
        onClose={cancelLora.closeDialog}
        onStopNow={() => cancelLora.dialogLora && void cancelLora.executeCancel(cancelLora.dialogLora, false)}
        onSaveAndStop={() => cancelLora.dialogLora && void cancelLora.executeCancel(cancelLora.dialogLora, true)}
      />

      <Modal
        open={showReproduceModal}
        onClose={() => setShowReproduceModal(false)}
        title="Reproduce Training"
        description="Create a new LoRA using this run's frozen config."
      >
        {reproduceError && <ModalError>{reproduceError}</ModalError>}
        <Input label="LoRA Name" value={reproduceName} onChange={(e) => setReproduceName(e.target.value)} />
        <Checkbox
          label="Enqueue immediately"
          checked={reproduceEnqueue}
          onChange={(e) => setReproduceEnqueue(e.target.checked)}
        />
        <ModalFooter>
          <Button variant="secondary" onClick={() => setShowReproduceModal(false)}>
            Cancel
          </Button>
          <Button onClick={() => void handleReproduce()} disabled={submitting}>
            {submitting ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}
            {submitting ? "Creating…" : "Create LoRA"}
          </Button>
        </ModalFooter>
      </Modal>
    </>
  );
}
