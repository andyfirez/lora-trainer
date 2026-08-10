import type { RunnableStatus } from "@/types";

const STATUS_CLASS: Record<RunnableStatus, string> = {
  draft: "status-pending",
  queued: "status-queued",
  running: "status-running",
  completed: "status-completed",
  failed: "status-failed",
  cancelled: "status-cancelled",
  orphan: "status-failed",
};

export default function StatusBadge({ status }: { status: RunnableStatus }) {
  return (
    <span className={`status-badge ${STATUS_CLASS[status]}`}>
      {status}
    </span>
  );
}
