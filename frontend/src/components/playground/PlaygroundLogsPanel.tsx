"use client";

import { useEffect, useRef, useState } from "react";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import { formSectionTitleClass } from "@/components/ui/FormSection";
import { samplingsApi } from "@/lib/api/samplings";
import type { SamplingResponse } from "@/types";

const LOG_STATUSES = new Set(["queued", "running", "completed", "failed", "cancelled"]);

function defaultLogsOpen(job: SamplingResponse | undefined): boolean {
  if (!job) return false;
  return job.status === "queued" || job.status === "running" || job.status === "failed";
}

interface PlaygroundLogsPanelProps {
  job: SamplingResponse | undefined;
}

export default function PlaygroundLogsPanel({ job }: PlaygroundLogsPanelProps) {
  const [logsOpen, setLogsOpen] = useState(false);
  const prevJobIdRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (job?.id !== prevJobIdRef.current) {
      prevJobIdRef.current = job?.id;
      setLogsOpen(defaultLogsOpen(job));
    }
  }, [job]);

  const showLogs = Boolean(job && LOG_STATUSES.has(job.status));

  return (
    <section className="border-t border-border p-3">
      <button
        type="button"
        onClick={() => setLogsOpen((open) => !open)}
        className={`${formSectionTitleClass} mb-0 w-full text-left flex items-center justify-between`}
      >
        Logs
        <span className="text-muted text-xs">{logsOpen ? "▲" : "▼"}</span>
      </button>
      {logsOpen && (
        <div className="mt-3 space-y-3">
          {showLogs && job ? (
            <LiveLogsPanel
              swrKey={`/samplings/${job.id}/logs`}
              fetcher={() => samplingsApi.getLogs(job.id, 500)}
              isRunning={job.status === "running"}
              showLogs={showLogs}
              status={job.status}
              highlightErrors
              hideTitle
            />
          ) : (
            <p className="text-sm text-muted">Generate to see logs…</p>
          )}
        </div>
      )}
    </section>
  );
}
