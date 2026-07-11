"use client";

import useSWR from "swr";
import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, Play, Square, Download, Loader2, Images } from "lucide-react";
import dynamic from "next/dynamic";
import { jobsApi } from "@/lib/api/jobs";
import { configsApi } from "@/lib/api/configs";
import { useSourceJobChildren } from "@/hooks/useSourceJobChildren";
import StatusBadge from "@/components/StatusBadge";
import TrainingJobPanel from "@/components/TrainingJobPanel";
import SamplingJobPanel from "@/components/SamplingJobPanel";
import TaggingJobPanel from "@/components/TaggingJobPanel";
import StopJobDialog from "@/components/StopJobDialog";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { ModalError } from "@/components/ui/Modal";
import { useCancelJob } from "@/hooks/useCancelJob";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  params: Promise<{ jobId: string }>;
}

export default function JobDetailPage({ params }: Props) {
  const { jobId } = use(params);
  const id = Number(jobId);
  const router = useRouter();
  const { data: job, isLoading, mutate } = useSWR(`/jobs/${id}`, () => jobsApi.get(id), {
    refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 2000),
  });
  const { hasActiveSamplingJob, mutate: mutateChildren } = useSourceJobChildren(
    job?.job_type === "training" ? id : null,
  );
  const [lossGraphRunKey, setLossGraphRunKey] = useState(0);
  const [samplingLoading, setSamplingLoading] = useState(false);
  const [samplingError, setSamplingError] = useState<string | null>(null);
  const prevJobStatusRef = useRef<string | null>(null);
  const {
    dialogJob,
    loading: cancelLoading,
    error: cancelError,
    canSaveCheckpoint,
    requestCancel,
    executeCancel,
    closeDialog,
  } = useCancelJob(() => void mutate());

  useEffect(() => {
    if (!job) return;
    const prevStatus = prevJobStatusRef.current;
    if (
      job.job_type === "training" &&
      job.status === "queued" &&
      prevStatus !== "queued" &&
      !job.training?.resume_checkpoint_path
    ) {
      setLossGraphRunKey((key) => key + 1);
    }
    prevJobStatusRef.current = job.status;
  }, [job]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }
  if (!job) return <div className="text-error">Job not found</div>;

  const isTraining = job.job_type === "training";
  const isTagging = job.job_type === "tagging";
  const showRunSampling =
    isTraining &&
    job.status === "completed" &&
    job.training?.sampling_config_id != null &&
    !hasActiveSamplingJob;

  const handleEnqueue = async () => {
    await jobsApi.enqueue(id);
    mutate();
  };
  const handleResume = async () => {
    await jobsApi.resume(id);
    mutate();
  };
  const handleCancel = () => {
    requestCancel(job);
  };
  const handleDownloadYaml = () => {
    const blob = new Blob([job.config_yaml], { type: "text/yaml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${job.name}.yaml`;
    a.click();
  };
  const handleRunSampling = async () => {
    const samplingConfigId = job.training?.sampling_config_id;
    if (samplingConfigId == null) return;
    setSamplingError(null);
    setSamplingLoading(true);
    try {
      const samplingJob = await configsApi.createJob(samplingConfigId, {
        name: `${job.name} sampling`,
        source_job_id: job.id,
        enqueue: true,
      });
      await mutateChildren();
      mutate();
      router.push(`/jobs/${samplingJob.id}`);
    } catch (err) {
      setSamplingError(err instanceof Error ? err.message : "Failed to start sampling");
    } finally {
      setSamplingLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/jobs" className="p-2 rounded-lg hover:bg-white/5 text-muted hover:text-text" aria-label="Back to jobs">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-text font-display">{job.name}</h1>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <StatusBadge status={job.status} />
            <span className="text-xs text-muted capitalize">{job.job_type}</span>
            {job.pid && <span className="text-xs text-muted">PID {job.pid}</span>}
            {job.config_id != null && (
              <Link href={`/configs/${job.config_id}`} className="text-xs text-accent hover:underline">
                Config #{job.config_id}
              </Link>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap shrink-0">
          {showRunSampling && (
            <Button variant="sampling" size="sm" onClick={() => void handleRunSampling()} disabled={samplingLoading}>
              {samplingLoading ? <Loader2 size={13} className="animate-spin" /> : <Images size={13} />}
              Run Sampling
            </Button>
          )}
          {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
            <Button variant="success" size="sm" onClick={() => void handleEnqueue()}>
              <Play size={13} /> Enqueue
            </Button>
          )}
          {(job.status === "failed" || job.status === "cancelled") && job.can_resume && isTraining && (
            <Button variant="primary" size="sm" onClick={() => void handleResume()} className="bg-running hover:bg-running/90">
              <Play size={13} /> Resume
            </Button>
          )}
          {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
            <Button variant="danger" size="sm" onClick={handleCancel} disabled={cancelLoading}>
              {cancelLoading ? <Loader2 size={13} className="animate-spin" /> : <Square size={13} />}{" "}
              {job.status === "running"
                ? isTraining
                  ? "Stop Training"
                  : isTagging
                    ? "Stop Tagging"
                    : "Stop Sampling"
                : "Cancel"}
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={handleDownloadYaml}>
            <Download size={13} /> YAML
          </Button>
        </div>
      </div>

      {samplingError && <ModalError>{samplingError}</ModalError>}
      {cancelError && !dialogJob && <ModalError>{cancelError}</ModalError>}

      <StopJobDialog
        open={dialogJob != null}
        jobName={dialogJob?.name ?? job.name}
        canSaveCheckpoint={canSaveCheckpoint}
        loading={cancelLoading}
        error={cancelError}
        onClose={closeDialog}
        onStopNow={() => dialogJob && void executeCancel(dialogJob, false)}
        onSaveAndStop={() => dialogJob && void executeCancel(dialogJob, true)}
      />

      {isTraining ? (
        <TrainingJobPanel job={job} lossGraphRunKey={lossGraphRunKey} />
      ) : isTagging ? (
        <TaggingJobPanel job={job} />
      ) : (
        <SamplingJobPanel job={job} />
      )}

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-muted">Config YAML</h2>
        <Card padding="none" className="overflow-hidden" style={{ height: 400 }}>
          <MonacoEditor
            height="100%"
            language="yaml"
            theme="vs-dark"
            value={job.config_yaml}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, scrollBeyondLastLine: false }}
          />
        </Card>
      </div>
    </div>
  );
}
