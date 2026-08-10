"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { ArrowLeft, Download, Loader2, Play, RotateCcw, Square } from "lucide-react";
import { lorasApi } from "@/lib/api/loras";
import { useCancelLora } from "@/hooks/useCancelLora";
import StatusBadge from "@/components/StatusBadge";
import StopJobDialog from "@/components/StopJobDialog";
import LoraRunPanel from "@/components/lora/LoraRunPanel";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { ModalError, ModalFooter } from "@/components/ui/Modal";
import Modal from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Checkbox from "@/components/ui/Checkbox";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  params: Promise<{ id: string }>;
}

export default function LoraDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const loraId = Number(idParam);
  const router = useRouter();
  const { data: lora, isLoading, mutate } = useSWR(`/loras/${loraId}`, () => lorasApi.get(loraId), {
    refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 2000),
  });

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

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }

  if (!lora) {
    return <div className="text-error py-20">LoRA not found</div>;
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/loras" className="p-2 rounded-lg hover:bg-white/5 text-muted hover:text-text" aria-label="Back to LoRAs">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-text font-display">{lora.name}</h1>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <StatusBadge status={lora.status} />
            <span className="text-muted truncate">{lora.base_model_name}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap shrink-0">
          {(lora.status === "draft" || lora.status === "failed" || lora.status === "cancelled" || lora.status === "orphan") && (
            <Button variant="success" size="sm" onClick={() => void handleEnqueue()}>
              <Play size={13} /> Enqueue
            </Button>
          )}
          {lora.can_resume && (
            <Button variant="secondary" size="sm" onClick={() => void handleResume()}>
              <RotateCcw size={13} /> Resume
            </Button>
          )}
          {(lora.status === "queued" || lora.status === "running") && (
            <Button variant="danger" size="sm" onClick={() => cancelLora.requestCancel(lora)}>
              <Square size={13} /> {lora.status === "running" ? "Stop" : "Cancel"}
            </Button>
          )}
          {lora.status === "completed" && (
            <a
              href={lorasApi.downloadWeightsUrl(loraId)}
              download
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-border bg-transparent hover:bg-white/5 text-text transition-colors"
            >
              <Download size={14} /> Weights
            </a>
          )}
          {lora.config_yaml && (
            <Button variant="secondary" size="sm" onClick={openReproduceModal}>
              <Play size={13} /> Reproduce
            </Button>
          )}
        </div>
      </div>

      <Card className="space-y-2 text-sm">
        <div className="text-muted break-all">Relative path: {lora.relative_path}</div>
        <div className="text-muted break-all">Weights: {lora.resolved_weights_path}</div>
        <div className="text-muted break-all">Work dir: {lora.resolved_work_dir}</div>
      </Card>

      <LoraRunPanel lora={lora} lossGraphRunKey={lora.id} />

      {lora.config_yaml && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-muted">Config YAML</h2>
          <Card padding="none" className="overflow-hidden" style={{ height: 400 }}>
            <MonacoEditor
              height="400px"
              defaultLanguage="yaml"
              theme="vs-dark"
              value={lora.config_yaml}
              options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
            />
          </Card>
        </div>
      )}

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
    </div>
  );
}
