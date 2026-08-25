"use client";

import useSWR from "swr";
import { useEffect, useRef } from "react";
import type { LogsResponse } from "@/types";

const ERROR_LINE_PATTERN = /\b(ERROR|Traceback|Exception|Error:)\b/i;

function isErrorLine(line: string): boolean {
  return ERROR_LINE_PATTERN.test(line);
}

interface Props {
  swrKey: string | null;
  fetcher: () => Promise<LogsResponse>;
  isRunning: boolean;
  showLogs: boolean;
  status: string;
  title?: string;
  highlightErrors?: boolean;
  hideTitle?: boolean;
}

export default function LiveLogsPanel({
  swrKey,
  fetcher,
  isRunning,
  showLogs,
  status,
  title = "Logs",
  highlightErrors = false,
  hideTitle = false,
}: Props) {
  const logRef = useRef<HTMLPreElement>(null);
  const prevRunningRef = useRef(isRunning);
  const { data, error, mutate } = useSWR(swrKey, fetcher, { refreshInterval: isRunning ? 2000 : 0 });

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

  const lines = data?.lines ?? [];
  const hasLines = lines.length > 0;

  return (
    <div className="space-y-2">
      {!hideTitle ? <h2 className="text-sm font-medium text-muted">{title}</h2> : null}
      {error ? (
        <p className="text-xs text-error">
          {error instanceof Error ? error.message : "Failed to load logs"}
        </p>
      ) : null}
      <pre
        ref={logRef}
        className="bg-surface rounded-xl border border-border p-4 text-xs font-mono overflow-auto whitespace-pre-wrap break-words"
        style={{ height: 320 }}
      >
        {highlightErrors && hasLines ? (
          lines.map((line, index) => (
            <span key={index} className={isErrorLine(line) ? "text-error" : "text-success"}>
              {line}
              {index < lines.length - 1 ? "\n" : ""}
            </span>
          ))
        ) : (
          <span className="text-success">{hasLines ? lines.join("\n") : "No logs yet…"}</span>
        )}
      </pre>
    </div>
  );
}
