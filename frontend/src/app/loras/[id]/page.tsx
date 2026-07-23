"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { ArrowLeft, Download, Loader2, Play } from "lucide-react";
import { lorasApi } from "@/lib/api/loras";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { ModalError, ModalFooter } from "@/components/ui/Modal";
import Modal from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Checkbox from "@/components/ui/Checkbox";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface Props {
  params: Promise<{ id: string }>;
}

export default function LoraDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const loraId = Number(idParam);
  const router = useRouter();
  const { data: lora, isLoading } = useSWR(`/loras/${loraId}`, () => lorasApi.get(loraId));
  const { data: samplesData } = useSWR(lora ? `/loras/${loraId}/samples` : null, () => lorasApi.getSamples(loraId));

  const [showReproduceModal, setShowReproduceModal] = useState(false);
  const [jobName, setJobName] = useState("");
  const [enqueue, setEnqueue] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [reproduceError, setReproduceError] = useState<string | null>(null);

  const openReproduceModal = () => {
    setJobName(lora ? `${lora.name} reproduce` : "");
    setEnqueue(true);
    setReproduceError(null);
    setShowReproduceModal(true);
  };

  const handleReproduce = async () => {
    if (!jobName.trim()) {
      setReproduceError("Job name is required");
      return;
    }
    setSubmitting(true);
    setReproduceError(null);
    try {
      const job = await lorasApi.reproduce(loraId, { name: jobName.trim(), enqueue });
      setShowReproduceModal(false);
      router.push(`/jobs/${job.id}`);
    } catch (err: unknown) {
      setReproduceError(err instanceof Error ? err.message : "Failed to reproduce training");
    } finally {
      setSubmitting(false);
    }
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

  const samples = samplesData?.samples ?? [];

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/loras" className="p-2 rounded-lg hover:bg-white/5 text-muted hover:text-text" aria-label="Back to LoRAs">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-text font-display">{lora.name}</h1>
          <p className="text-muted mt-1 truncate">{lora.base_model_name}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <a
            href={lorasApi.downloadWeightsUrl(loraId)}
            download
            className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-border bg-transparent hover:bg-white/5 text-text transition-colors"
          >
            <Download size={14} /> Weights
          </a>
          {lora.config_yaml && (
            <Button variant="success" size="sm" onClick={openReproduceModal}>
              <Play size={14} /> Reproduce
            </Button>
          )}
        </div>
      </div>

      <Card className="space-y-2 text-sm">
        {lora.job_id != null && (
          <div>
            <span className="text-muted">Training job:</span>{" "}
            <Link href={`/jobs/${lora.job_id}`} className="text-accent hover:underline">
              #{lora.job_id}
            </Link>
          </div>
        )}
        {lora.config_id != null && (
          <div>
            <span className="text-muted">Source config:</span>{" "}
            <Link href={`/trainings/${lora.config_id}`} className="text-accent hover:underline">
              #{lora.config_id}
            </Link>
          </div>
        )}
        <div className="text-muted break-all">Relative path: {lora.relative_path}</div>
        <div className="text-muted break-all">Weights: {lora.resolved_weights_path}</div>
        <div className="text-muted break-all">Work dir: {lora.resolved_work_dir}</div>
      </Card>

      {samples.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-muted">Samples</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {samples.map((sample) => (
              <a key={sample.path} href={`${API_BASE_URL}${sample.url}`} target="_blank" rel="noreferrer">
                <img
                  src={`${API_BASE_URL}${sample.url}`}
                  alt={sample.filename}
                  className="rounded-lg border border-border object-cover aspect-square"
                />
              </a>
            ))}
          </div>
        </div>
      )}

      {lora.config_yaml && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-muted">Frozen Config YAML</h2>
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

      <Modal
        open={showReproduceModal}
        onClose={() => setShowReproduceModal(false)}
        title="Reproduce Training"
        description="Create a new training job using this LoRA's frozen config."
      >
        {reproduceError && <ModalError>{reproduceError}</ModalError>}
        <Input label="Job Name" value={jobName} onChange={(e) => setJobName(e.target.value)} />
        <Checkbox label="Enqueue immediately" checked={enqueue} onChange={(e) => setEnqueue(e.target.checked)} />
        <ModalFooter>
          <Button variant="secondary" onClick={() => setShowReproduceModal(false)}>
            Cancel
          </Button>
          <Button onClick={() => void handleReproduce()} disabled={submitting}>
            {submitting ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}
            {submitting ? "Creating…" : "Create Job"}
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
