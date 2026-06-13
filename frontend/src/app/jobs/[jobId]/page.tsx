"use client";

import useSWR from "swr";
import { use, useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Play, Square, Download, Pencil } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import StatusBadge from "@/components/StatusBadge";
import JobProgressBar from "@/components/JobProgressBar";
import JobLossGraph from "@/components/JobLossGraph";
import { Loader2 } from "lucide-react";
import dynamic from "next/dynamic";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  params: Promise<{ jobId: string }>;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

function LiveLogs({
  jobId,
  isRunning,
  showLogs,
  jobStatus,
}: {
  jobId: number;
  isRunning: boolean;
  showLogs: boolean;
  jobStatus: string;
}) {
  const logRef = useRef<HTMLPreElement>(null);
  const prevRunningRef = useRef(isRunning);
  const { data, mutate } = useSWR(
    showLogs ? `/jobs/${jobId}/logs` : null,
    () => jobsApi.getLogs(jobId, 500),
    { refreshInterval: isRunning ? 2000 : 0 },
  );

  useEffect(() => {
    if (prevRunningRef.current && !isRunning && showLogs) {
      if (jobStatus === "cancelled") {
        void mutate({ lines: [] }, { revalidate: false });
      } else {
        void mutate();
      }
    }
    prevRunningRef.current = isRunning;
  }, [isRunning, showLogs, mutate, jobStatus]);

  useEffect(() => {
    if (jobStatus === "cancelled") {
      void mutate({ lines: [] }, { revalidate: false });
    }
  }, [jobStatus, mutate]);

  useEffect(() => {
    if (logRef.current && (isRunning || data?.lines.length)) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [data, isRunning]);

  const text = data?.lines.join("\n") ?? "";

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium text-[var(--muted)]">Training Logs</h2>
      <pre
        ref={logRef}
        className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 text-xs text-green-300 font-mono overflow-auto whitespace-pre-wrap break-words"
        style={{ height: 320 }}
      >
        {text || "No logs yet…"}
      </pre>
    </div>
  );
}

export default function JobDetailPage({ params }: Props) {
  const { jobId } = use(params);
  const id = Number(jobId);
  const { data: job, isLoading, mutate } = useSWR(`/jobs/${id}`, () => jobsApi.get(id), { refreshInterval: 2000 });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-[var(--muted)] py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }
  if (!job) return <div className="text-red-400">Job not found</div>;

  const cachePct = progressPercent(job.cache_progress_step, job.cache_progress_total);
  const trainPct = progressPercent(job.progress_step, job.progress_total);
  const samplingPct =
    job.sampling_step != null && job.sampling_total != null && job.sampling_total > 0
      ? Math.round((job.sampling_step / job.sampling_total) * 100)
      : null;
  const isRunning = job.status === "running";
  const showLogs = isRunning || job.status === "completed" || job.status === "failed";
  const showLossGraph = showLogs;

  const handleEnqueue = async () => { await jobsApi.enqueue(id); mutate(); };
  const handleCancel = async () => { await jobsApi.cancel(id); mutate(); };
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
          <div className="flex items-center gap-3 mt-1">
            <StatusBadge status={job.status} />
            {job.pid && <span className="text-xs text-[var(--muted)]">PID {job.pid}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
            <button onClick={handleEnqueue} className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 text-white rounded-lg px-3 py-1.5 text-sm">
              <Play size={13} /> Enqueue
            </button>
          )}
          {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
            <button onClick={handleCancel} className="flex items-center gap-1.5 bg-red-700 hover:bg-red-600 text-white rounded-lg px-3 py-1.5 text-sm">
              <Square size={13} /> {job.status === "running" ? "Stop Training" : "Cancel"}
            </button>
          )}
          <button onClick={handleDownloadYaml} className="flex items-center gap-1.5 border border-[var(--border)] hover:bg-white/5 text-[var(--muted)] hover:text-white rounded-lg px-3 py-1.5 text-sm">
            <Download size={13} /> YAML
          </button>
          {job.status !== "running" && (
            <Link href={`/jobs/new?edit=${id}`} className="flex items-center gap-1.5 border border-[var(--border)] hover:bg-white/5 text-[var(--muted)] hover:text-white rounded-lg px-3 py-1.5 text-sm">
              <Pencil size={13} /> Edit
            </Link>
          )}
        </div>
      </div>

      {isRunning && cachePct != null && (
        <JobProgressBar
          title="Caching Progress"
          step={job.cache_progress_step}
          total={job.cache_progress_total}
          percent={cachePct}
          active={isRunning}
          barClassName="bg-amber-500"
          headerRight={
            <span className="text-[var(--muted)]">
              {job.cache_progress_step} / {job.cache_progress_total} ({cachePct}%)
            </span>
          }
        />
      )}

      {isRunning && (job.sampling_status ? (
        <JobProgressBar
          title={job.sampling_status}
          step={job.sampling_step}
          total={job.sampling_total}
          percent={samplingPct ?? 0}
          active={isRunning}
          barClassName="bg-purple-500"
          showSpinner
          showBar={samplingPct != null}
          headerRight={
            samplingPct != null ? (
              <span className="text-[var(--muted)]">
                step {job.sampling_step} / {job.sampling_total} ({samplingPct}%)
              </span>
            ) : undefined
          }
        />
      ) : trainPct != null ? (
        <JobProgressBar
          title="Training Progress"
          step={job.progress_step}
          total={job.progress_total}
          percent={trainPct}
          active={isRunning}
          headerRight={
            <span className="text-[var(--muted)]">
              {job.progress_epoch != null && job.progress_epoch > 0 && job.progress_epoch_total != null && (
                <>epoch {job.progress_epoch}/{job.progress_epoch_total} · </>
              )}
              step {job.progress_step} / {job.progress_total} ({trainPct}%)
            </span>
          }
          footer={
            (job.progress_loss != null || job.progress_avr_loss != null) ? (
              <div className="flex gap-4 text-xs text-[var(--muted)]">
                {job.progress_loss != null && <span>loss: <span className="text-white">{job.progress_loss.toFixed(4)}</span></span>}
                {job.progress_avr_loss != null && <span>avr_loss: <span className="text-white">{job.progress_avr_loss.toFixed(4)}</span></span>}
              </div>
            ) : undefined
          }
        />
      ) : null)}

      {showLossGraph && <JobLossGraph jobId={id} isActive={isRunning} />}

      {showLogs && <LiveLogs jobId={id} isRunning={isRunning} showLogs={showLogs} jobStatus={job.status} />}

      {job.status === "failed" && job.error_message && (
        <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 text-sm">
          <strong>Error:</strong> {job.error_message}
        </div>
      )}

      {job.output_path && (
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4">
          <div className="text-xs text-[var(--muted)] mb-1">Output</div>
          <code className="text-green-400 text-sm">{job.output_path}</code>
        </div>
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
