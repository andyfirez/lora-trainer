"use client";

import useSWR from "swr";
import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Play, Square, Download, Pencil, Plus, X, Sparkles } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { samplingRunsApi } from "@/lib/api/samplingRuns";
import StatusBadge from "@/components/StatusBadge";
import JobProgressBar from "@/components/JobProgressBar";
import JobLossGraph from "@/components/JobLossGraph";
import PathInput from "@/components/PathInput";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import { Loader2 } from "lucide-react";
import dynamic from "next/dynamic";
import type { SamplingRun } from "@/types";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface Props {
  params: Promise<{ jobId: string }>;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

function SamplingRunSamples({ run }: { run: SamplingRun }) {
  const { data } = useSWR(
    run.status === "completed" ? `/sampling-runs/${run.id}/samples` : null,
    () => samplingRunsApi.getSamples(run.id),
  );

  if (!data?.samples.length) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
      {data.samples.map((sample) => (
        <a key={sample.path} href={`${API_BASE_URL}${sample.url}`} target="_blank" rel="noreferrer">
          <img
            src={`${API_BASE_URL}${sample.url}`}
            alt={sample.filename}
            className="rounded-lg border border-[var(--border)] object-cover aspect-square"
          />
        </a>
      ))}
    </div>
  );
}

function SamplingRunsPanel({ jobId }: { jobId: number }) {
  const { data: runs, mutate } = useSWR(`/jobs/${jobId}/sampling-runs`, () => jobsApi.listSamplingRuns(jobId), {
    refreshInterval: (latest) => (latest?.some((run) => run.status === "running") ? 1000 : 2000),
  });
  const [loraPaths, setLoraPaths] = useState<string[]>([""]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updatePath = (index: number, value: string) => {
    setLoraPaths((paths) => paths.map((path, i) => (i === index ? value : path)));
  };
  const addPath = () => setLoraPaths((paths) => [...paths, ""]);
  const removePath = (index: number) => {
    setLoraPaths((paths) => (paths.length === 1 ? [""] : paths.filter((_, i) => i !== index)));
  };
  const createSamplingRun = async () => {
    const selectedPaths = loraPaths.map((path) => path.trim()).filter(Boolean);
    if (!selectedPaths.length) {
      setError("Select at least one LoRA file.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await jobsApi.createSamplingRun(jobId, { lora_paths: selectedPaths });
      setLoraPaths([""]);
      await mutate();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create sampling run");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div id="sampling" className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 space-y-4">
      <div>
        <h2 className="text-sm font-medium text-white">Sampling</h2>
        <p className="text-xs text-[var(--muted)] mt-1">
          Run sampling after training for one or more selected LoRA files.
        </p>
      </div>

      <div className="space-y-3">
        {loraPaths.map((path, index) => (
          <div key={index} className="flex gap-2 items-end">
            <div className="flex-1">
              <PathInput
                label={`LoRA ${index + 1}`}
                value={path}
                onChange={(value) => updatePath(index, value)}
                pickerTitle="Select LoRA file"
                kind="model"
                placeholder="D:\\loras\\model_epoch1.safetensors"
              />
            </div>
            <button
              type="button"
              onClick={() => removePath(index)}
              className="mb-0.5 p-2 rounded-lg border border-[var(--border)] text-[var(--muted)] hover:bg-white/5 hover:text-white"
              title="Remove LoRA"
            >
              <X size={16} />
            </button>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={addPath}
            className="flex items-center gap-1.5 border border-[var(--border)] hover:bg-white/5 text-[var(--muted)] hover:text-white rounded-lg px-3 py-1.5 text-sm"
          >
            <Plus size={13} /> Add LoRA
          </button>
          <button
            type="button"
            onClick={() => void createSamplingRun()}
            disabled={submitting}
            className="flex items-center gap-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {submitting ? <Loader2 className="animate-spin" size={13} /> : <Play size={13} />} Run Sampling
          </button>
        </div>
        {error && <div className="text-xs text-red-400">{error}</div>}
      </div>

      {!!runs?.length && (
        <div className="space-y-3">
          {runs.map((run) => {
            const percent =
              run.progress_step != null && run.progress_total != null && run.progress_total > 0
                ? Math.round((run.progress_step / run.progress_total) * 100)
                : null;
            return (
              <div key={run.id} className="rounded-lg border border-[var(--border)] p-3 space-y-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <Link
                    href={`/sampling-runs/${run.id}`}
                    className="font-medium text-white text-sm hover:text-purple-400"
                  >
                    {run.name}
                  </Link>
                  <StatusBadge status={run.status} />
                  {run.pid && <span className="text-xs text-[var(--muted)]">PID {run.pid}</span>}
                  {run.status === "failed" && (
                    <Link href={`/sampling-runs/${run.id}`} className="text-xs text-red-400 hover:underline">
                      View logs
                    </Link>
                  )}
                </div>
                {run.progress_status && (
                  <JobProgressBar
                    title={run.progress_status}
                    step={run.progress_step}
                    total={run.progress_total}
                    percent={percent ?? 0}
                    active={run.status === "running"}
                    barClassName="bg-purple-500"
                    showSpinner={run.status === "running"}
                    showBar={percent != null}
                  />
                )}
                <div className="text-xs text-[var(--muted)]">
                  {run.lora_paths.length} LoRA file(s)
                  {run.output_path ? <> · output <code className="text-green-400">{run.output_path}</code></> : null}
                </div>
                {run.error_message && <div className="text-xs text-red-400">{run.error_message}</div>}
                <SamplingRunSamples run={run} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function JobDetailPage({ params }: Props) {
  const { jobId } = use(params);
  const id = Number(jobId);
  const { data: job, isLoading, mutate } = useSWR(`/jobs/${id}`, () => jobsApi.get(id), { refreshInterval: 2000 });
  const [lossGraphRunKey, setLossGraphRunKey] = useState(0);
  const prevJobStatusRef = useRef<string | null>(null);

  useEffect(() => {
    if (!job) {
      return;
    }
    const prevStatus = prevJobStatusRef.current;
    if (job.status === "queued" && prevStatus !== "queued" && !job.resume_checkpoint_path) {
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

  const trainStep = job.progress_step ?? 0;
  const trainPct =
    job.progress_total != null && job.progress_total > 0
      ? progressPercent(trainStep, job.progress_total)
      : null;
  const samplingPct =
    job.sampling_step != null && job.sampling_total != null && job.sampling_total > 0
      ? Math.round((job.sampling_step / job.sampling_total) * 100)
      : null;
  const isRunning = job.status === "running";
  const showLogs = isRunning || job.status === "completed" || job.status === "failed" || job.status === "cancelled";
  const showLossGraph = showLogs;

  const handleEnqueue = async () => { await jobsApi.enqueue(id); mutate(); };
  const handleResume = async () => { await jobsApi.resume(id); mutate(); };
  const handleCancel = async () => {
    if (job.status === "running") {
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
          <div className="flex items-center gap-3 mt-1">
            <StatusBadge status={job.status} />
            {job.pid && <span className="text-xs text-[var(--muted)]">PID {job.pid}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <a
            href="#sampling"
            className="flex items-center gap-1.5 bg-purple-700 hover:bg-purple-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            <Sparkles size={13} /> Sampling
          </a>
          {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
            <button onClick={handleEnqueue} className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 text-white rounded-lg px-3 py-1.5 text-sm">
              <Play size={13} /> Enqueue
            </button>
          )}
          {(job.status === "failed" || job.status === "cancelled") && job.can_resume && (
            <button onClick={handleResume} className="flex items-center gap-1.5 bg-blue-700 hover:bg-blue-600 text-white rounded-lg px-3 py-1.5 text-sm">
              <Play size={13} /> Resume
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

      {isRunning && job.sampling_status != null && (
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
      )}

      {isRunning && trainPct != null && (
        <JobProgressBar
          title="Training Progress"
          step={trainStep}
          total={job.progress_total}
          percent={trainPct}
          active={isRunning}
          headerRight={
            <span className="text-[var(--muted)]">
              {job.progress_epoch != null && job.progress_epoch > 0 && job.progress_epoch_total != null && (
                <>epoch {job.progress_epoch}/{job.progress_epoch_total} · </>
              )}
              step {trainStep} / {job.progress_total} ({trainPct}%)
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
      )}

      {showLossGraph && (
        <JobLossGraph jobId={id} isActive={isRunning} resetKey={String(lossGraphRunKey)} />
      )}

      <SamplingRunsPanel jobId={id} />

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/jobs/${id}/logs`}
          fetcher={() => jobsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={job.status}
          title="Training Logs"
        />
      )}

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
