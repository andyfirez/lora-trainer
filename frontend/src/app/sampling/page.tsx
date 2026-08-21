"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { Loader2, PlusCircle, Play, X } from "lucide-react";
import { samplingsApi } from "@/lib/api/samplings";
import { useCancelRunnable } from "@/hooks/useCancelRunnable";
import PageHeader from "@/components/ui/PageHeader";
import Tabs from "@/components/ui/Tabs";
import Card from "@/components/ui/Card";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import StopJobDialog from "@/components/StopJobDialog";
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from "@/components/ui/Table";
import type { SamplingResponse } from "@/types";
import { canCancel, canEnqueue, cancelActionLabel } from "@/lib/runnableStatus";

const ACTIVE_STATUSES = new Set(["draft", "queued", "running"]);

type TabValue = "active" | "completed";

export default function SamplingPage() {
  const [tab, setTab] = useState<TabValue>("active");
  const { data: samplings, isLoading, mutate } = useSWR("/samplings", () => samplingsApi.list(), {
    refreshInterval: (latest) => (latest?.some((s) => s.status === "running") ? 1000 : 5000),
  });
  const cancelSampling = useCancelRunnable<SamplingResponse>(samplingsApi, () => void mutate(), {
    errorMessage: "Failed to stop sampling",
  });

  const filtered = useMemo(() => {
    const all = samplings ?? [];
    return tab === "active"
      ? all.filter((s) => ACTIVE_STATUSES.has(s.status))
      : all.filter((s) => !ACTIVE_STATUSES.has(s.status));
  }, [samplings, tab]);

  const handleEnqueue = async (sampling: SamplingResponse) => {
    await samplingsApi.enqueue(sampling.id);
    mutate();
  };

  return (
    <>
      <div className="space-y-6">
        <PageHeader
          title="Sampling"
          description="Generate sample images from trained LoRAs"
          actions={
            <Link
              href="/sampling/new"
              className="inline-flex items-center gap-2 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              <PlusCircle size={15} />
              New sampling
            </Link>
          }
        />

        <Tabs
          tabs={[
            { value: "active", label: "Active" },
            { value: "completed", label: "Completed" },
          ]}
          value={tab}
          onChange={setTab}
        />

        {isLoading ? (
          <div className="flex items-center justify-center py-20 text-muted">
            <Loader2 className="animate-spin mr-2" size={18} /> Loading…
          </div>
        ) : !filtered.length ? (
          <Card className="text-center py-20 text-muted">
            {tab === "active" ? (
              <>
                No active sampling runs.{" "}
                <Link href="/sampling/new" className="text-accent hover:underline">
                  Start one
                </Link>
              </>
            ) : (
              "No completed sampling runs yet."
            )}
          </Card>
        ) : (
          <Table>
            <TableHead>
              <tr>
                <TableHeader>Name</TableHeader>
                <TableHeader>Status</TableHeader>
                <TableHeader>Progress</TableHeader>
                <TableHeader>Created</TableHeader>
                <TableHeader className="text-right">Actions</TableHeader>
              </tr>
            </TableHead>
            <TableBody>
              {filtered.map((sampling) => {
                const progress =
                  sampling.progress_step != null && sampling.progress_total != null && sampling.progress_total > 0
                    ? Math.round((sampling.progress_step / sampling.progress_total) * 100)
                    : null;
                return (
                  <TableRow key={sampling.id}>
                    <TableCell>
                      <Link href={`/sampling/${sampling.id}`} className="text-text font-medium hover:text-sampling">
                        {sampling.name}
                      </Link>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={sampling.status} />
                    </TableCell>
                    <TableCell>
                      {sampling.status === "running" && progress != null ? (
                        <div className="space-y-1">
                          {sampling.progress_status && (
                            <div className="text-xs text-muted truncate max-w-[200px]">{sampling.progress_status}</div>
                          )}
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-border rounded-full h-1.5 w-24">
                              <div
                                className="bg-sampling h-1.5 rounded-full transition-all"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                            <span className="text-muted text-xs">{progress}%</span>
                          </div>
                          <ProgressTimingInfo
                            step={sampling.progress_step}
                            total={sampling.progress_total}
                            active
                            elapsedSeconds={sampling.elapsed_seconds}
                            compact
                          />
                        </div>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-muted">{new Date(sampling.created_at).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        {canEnqueue(sampling.status) && (
                          <button
                            onClick={() => void handleEnqueue(sampling)}
                            title="Enqueue"
                            className="p-1.5 rounded hover:bg-white/10 text-success hover:text-success"
                          >
                            <Play size={14} />
                          </button>
                        )}
                        {canCancel(sampling.status) && (
                          <button
                            onClick={() => cancelSampling.requestCancel(sampling)}
                            title={cancelActionLabel(sampling.status)}
                            className="p-1.5 rounded hover:bg-white/10 text-error hover:text-error"
                          >
                            <X size={14} />
                          </button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>

      <StopJobDialog
        open={cancelSampling.dialogRunnable != null}
        jobName={cancelSampling.dialogRunnable?.name ?? ""}
        canSaveCheckpoint={cancelSampling.canSaveCheckpoint}
        loading={cancelSampling.loading}
        error={cancelSampling.error}
        onClose={cancelSampling.closeDialog}
        onStopNow={() =>
          cancelSampling.dialogRunnable &&
          void cancelSampling.executeCancel(cancelSampling.dialogRunnable, false)
        }
        onSaveAndStop={() =>
          cancelSampling.dialogRunnable &&
          void cancelSampling.executeCancel(cancelSampling.dialogRunnable, true)
        }
      />
    </>
  );
}
