"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, Play, Loader2, Copy } from "lucide-react";
import { trainingsApi } from "@/lib/api/trainings";
import ConfigForm from "@/components/ConfigForm";
import Button from "@/components/ui/Button";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Checkbox from "@/components/ui/Checkbox";

interface Props {
  params: Promise<{ id: string }>;
}

export default function TrainingDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const configId = Number(idParam);
  const router = useRouter();
  const { data: config, isLoading, mutate } = useSWR(`/trainings/${configId}`, () => trainingsApi.get(configId));

  const [showRunModal, setShowRunModal] = useState(false);
  const [jobName, setJobName] = useState("");
  const [enqueue, setEnqueue] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [cloning, setCloning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const openRunModal = () => {
    setJobName(config?.name ?? "");
    setEnqueue(true);
    setRunError(null);
    setShowRunModal(true);
  };

  const handleClone = async () => {
    setCloning(true);
    try {
      const cloned = await trainingsApi.clone(configId);
      router.push(`/trainings/${cloned.id}`);
    } finally {
      setCloning(false);
    }
  };

  const handleRunJob = async () => {
    if (!jobName.trim()) {
      setRunError("Job name is required");
      return;
    }
    setSubmitting(true);
    setRunError(null);
    try {
      const job = await trainingsApi.createJob(configId, { name: jobName.trim(), enqueue });
      setShowRunModal(false);
      router.push(`/jobs/${job.id}`);
    } catch (err: unknown) {
      setRunError(err instanceof Error ? err.message : "Failed to create job");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSaved = () => {
    void mutate();
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }

  if (!config) {
    return <div className="text-error py-20">Training config not found</div>;
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link
          href="/trainings"
          className="p-2 rounded-lg hover:bg-white/5 text-muted hover:text-text"
          aria-label="Back to trainings"
        >
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-text font-display">{config.name}</h1>
          <p className="text-muted mt-1">Training config</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="secondary" onClick={() => void handleClone()} disabled={cloning}>
            {cloning ? <Loader2 className="animate-spin" size={14} /> : <Copy size={14} />}
            {cloning ? "Duplicating…" : "Duplicate"}
          </Button>
          <Button variant="success" onClick={openRunModal}>
            <Play size={14} /> Run Job
          </Button>
        </div>
      </div>

      <ConfigForm
        key={config.updated_at}
        configType="training"
        configId={configId}
        initialName={config.name}
        initialDescription={config.description ?? ""}
        initialYaml={config.config_yaml}
        saveRedirectBase="/trainings"
        onSaved={handleSaved}
      />

      <Modal
        open={showRunModal}
        onClose={() => setShowRunModal(false)}
        title="Run Job"
        description="Create a new training job from the current config and optionally add it to the queue."
      >
        {runError && <ModalError>{runError}</ModalError>}
        <Input label="Job Name" value={jobName} onChange={(e) => setJobName(e.target.value)} />
        <Checkbox label="Enqueue immediately" checked={enqueue} onChange={(e) => setEnqueue(e.target.checked)} />
        <ModalFooter>
          <Button variant="secondary" onClick={() => setShowRunModal(false)}>
            Cancel
          </Button>
          <Button onClick={() => void handleRunJob()} disabled={submitting}>
            {submitting ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}
            {submitting ? "Creating…" : "Create Job"}
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
