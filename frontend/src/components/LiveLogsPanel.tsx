"use client";

import useSWR from "swr";
import { useEffect, useRef } from "react";

interface LogsResponse {
  lines: string[];
}

interface Props {
  swrKey: string | null;
  fetcher: () => Promise<LogsResponse>;
  isRunning: boolean;
  showLogs: boolean;
  status: string;
  title?: string;
}

export default function LiveLogsPanel({
  swrKey,
  fetcher,
  isRunning,
  showLogs,
  status,
  title = "Logs",
}: Props) {
  const logRef = useRef<HTMLPreElement>(null);
  const prevRunningRef = useRef(isRunning);
  const { data, mutate } = useSWR(swrKey, fetcher, { refreshInterval: isRunning ? 2000 : 0 });

  useEffect(() => {
    if (prevRunningRef.current && !isRunning && showLogs) {
      void mutate();
    }
    prevRunningRef.current = isRunning;
  }, [isRunning, showLogs, mutate, status]);

  useEffect(() => {
    if (logRef.current && (isRunning || data?.lines.length)) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [data, isRunning]);

  const text = data?.lines.join("\n") ?? "";

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium text-[var(--muted)]">{title}</h2>
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
