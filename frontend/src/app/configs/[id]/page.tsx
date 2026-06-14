"use client";

import { use, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, Play, Loader2, X, Copy } from "lucide-react";
import { parse as yamlParse } from "yaml";
import { configsApi } from "@/lib/api/configs";
import ConfigForm from "@/components/ConfigForm";

interface Props {
  params: Promise<{ id: string }>;
}

export default function ConfigDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const configId = Number(idParam);
  const router = useRouter();
  const { data: config, isLoading } = useSWR(`/configs/${configId}`, () => configsApi.get(configId));

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
      const cloned = await configsApi.clone(configId);
      router.push(`/configs/${cloned.id}`);
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
      let body: Parameters<typeof configsApi.createJob>[1] = { name: jobName.trim(), enqueue };
      if (config?.config_type === "sampling") {
        const parsed = yamlParse(config.config_yaml) as { lora_paths?: string[] };
        const loraPaths = (parsed.lora_paths ?? []).map((p) => p.trim()).filter(Boolean);
        if (loraPaths.length) {
          body = { ...body, lora_paths: loraPaths };
        }
      }
      const job = await configsApi.createJob(configId, body);
      setShowRunModal(false);
      router.push(`/jobs/${job.id}`);
    } catch (err: unknown) {
      setRunError(err instanceof Error ? err.message : "Failed to create job");
    } finally {
      setSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-[var(--muted)] py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }

  if (!config) {
    return <div className="text-red-400 py-20">Config not found</div>;
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/configs" className="p-2 rounded-lg hover:bg-white/5 text-[var(--muted)] hover:text-white">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{config.name}</h1>
          <p className="text-[var(--muted)] mt-1 capitalize">{config.config_type} config</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void handleClone()}
            disabled={cloning}
            className="flex items-center gap-1.5 border border-[var(--border)] hover:bg-white/5 disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm font-medium"
          >
            {cloning ? <Loader2 className="animate-spin" size={14} /> : <Copy size={14} />}
            {cloning ? "Duplicating…" : "Duplicate"}
          </button>
          <button
            onClick={openRunModal}
            className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 text-white rounded-lg px-4 py-2 text-sm font-medium"
          >
            <Play size={14} /> Run Job
          </button>
        </div>
      </div>

      <ConfigForm
        configType={config.config_type}
        configId={configId}
        initialName={config.name}
        initialDescription={config.description ?? ""}
        initialYaml={config.config_yaml}
      />

      {showRunModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-6 w-full max-w-md space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Run Job</h2>
              <button
                onClick={() => setShowRunModal(false)}
                className="p-1 rounded hover:bg-white/10 text-[var(--muted)] hover:text-white"
              >
                <X size={18} />
              </button>
            </div>
            <p className="text-sm text-[var(--muted)]">
              Create a new job from this config and optionally add it to the queue.
            </p>
            {runError && (
              <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-3 py-2 text-sm">
                {runError}
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-[var(--muted)] mb-1">Job Name</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-white focus:outline-none focus:border-[var(--accent)]"
              />
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                className="w-4 h-4 rounded accent-[var(--accent)]"
                checked={enqueue}
                onChange={(e) => setEnqueue(e.target.checked)}
              />
              <span className="text-sm text-white">Enqueue immediately</span>
            </label>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowRunModal(false)}
                className="px-4 py-2 text-sm text-[var(--muted)] hover:text-white border border-[var(--border)] rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={() => void handleRunJob()}
                disabled={submitting}
                className="flex items-center gap-1.5 bg-[var(--accent)] hover:bg-[var(--accent-hover)] disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm font-medium"
              >
                {submitting ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}
                {submitting ? "Creating…" : "Create Job"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
