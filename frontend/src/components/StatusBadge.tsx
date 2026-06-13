import type { JobStatus } from "@/types";

const STATUS_CLASS: Record<JobStatus, string> = {
  pending: "status-pending",
  queued: "status-queued",
  running: "status-running",
  completed: "status-completed",
  failed: "status-failed",
  cancelled: "status-cancelled",
};

export default function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`status-badge ${STATUS_CLASS[status]}`}>
      {status}
    </span>
  );
}
