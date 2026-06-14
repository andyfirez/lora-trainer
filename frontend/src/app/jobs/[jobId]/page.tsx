"use client";

import useSWR from "swr";
import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Play, Square, Download, Loader2 } from "lucide-react";
import dynamic from "next/dynamic";
import { jobsApi } from "@/lib/api/jobs";
import StatusBadge from "@/components/StatusBadge";
import TrainingJobPanel from "@/components/TrainingJobPanel";
import SamplingJobPanel from "@/components/SamplingJobPanel";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  params: Promise<{ jobId: string }>;
}

export default function JobDetailPage({ params }: Props) {
  const { jobId } = use(params);
  const id = Number(jobId);
  const { data: job, isLoading, mutate } = useSWR(`/jobs/${id}`, () => jobsApi.get(id), {
    refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 2000),
  });
  const [lossGraphRunKey, setLossGraphRunKey] = useState(0);
  const prevJobStatusRef = useRef<string | null>(null);

  useEffect(() => {
    if (!job) {
      return;
    }
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
      <div className="flex items-center gap-2 text-[var(--muted)] py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }
  if (!job) return <div className="text-red-400">Job not found</div>;

  const isTraining = job.job_type === "training";

  const handleEnqueue = async () => {
    await jobsApi.enqueue(id);
    mutate();
  };
  const handleResume = async () => {
    await jobsApi.resume(id);
    mutate();
  };
  const handleCancel = async () => {
    if (job.status === "running" && isTraining) {
      const saveCheckpoint = window.confirm("Save checkpoint before stopping this job?");
      await jobsApi.cancel(id, saveCheckpoint);
    } else {
      await jobsApi.cancel(id);
    }
    mutate();
  };
  const handleDownloadYaml = () => {
    const blob = new Blob([job.config_yaml], { type: "text/yaml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${job.name}.yaml`;
    a.click();
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/jobs" className="p-2 rounded-lg hover:bg-white/5 text-[var(--muted)] hover:text-white">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{job.name}</h1>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <StatusBadge status={job.status} />
            <span className="text-xs text-[var(--muted)] capitalize">{job.job_type}</span>
            {job.pid && <span className="text-xs text-[var(--muted)]">PID {job.pid}</span>}
            {job.config_id != null && (
              <Link href={`/configs/${job.config_id}`} className="text-xs text-[var(--accent)] hover:underline">
                Config #{job.config_id}
              </Link>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
            <button
              onClick={() => void handleEnqueue()}
              className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 text-white rounded-lg px-3 py-1.5 text-sm"
            >
              <Play size={13} /> Enqueue
            </button>
          )}
          {(job.status === "failed" || job.status === "cancelled") && job.can_resume && isTraining && (
            <button
              onClick={() => void handleResume()}
              className="flex items-center gap-1.5 bg-blue-700 hover:bg-blue-600 text-white rounded-lg px-3 py-1.5 text-sm"
            >
              <Play size={13} /> Resume
            </button>
          )}
          {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
            <button
              onClick={() => void handleCancel()}
              className="flex items-center gap-1.5 bg-red-700 hover:bg-red-600 text-white rounded-lg px-3 py-1.5 text-sm"
            >
              <Square size={13} /> {job.status === "running" ? (isTraining ? "Stop Training" : "Stop Sampling") : "Cancel"}
            </button>
          )}
          <button
            onClick={handleDownloadYaml}
            className="flex items-center gap-1.5 border border-[var(--border)] hover:bg-white/5 text-[var(--muted)] hover:text-white rounded-lg px-3 py-1.5 text-sm"
          >
            <Download size={13} /> YAML
          </button>
        </div>
      </div>

      {isTraining ? (
        <TrainingJobPanel job={job} lossGraphRunKey={lossGraphRunKey} />
      ) : (
        <SamplingJobPanel job={job} />
      )}

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-[var(--muted)]">Config YAML</h2>
        <div className="rounded-xl overflow-hidden border border-[var(--border)]" style={{ height: 400 }}>
          <MonacoEditor
            height="100%"
            language="yaml"
            theme="vs-dark"
            value={job.config_yaml}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, scrollBeyondLastLine: false }}
          />
        </div>
      </div>
    </div>
  );
}
