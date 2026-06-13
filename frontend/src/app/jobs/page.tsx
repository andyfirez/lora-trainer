import Link from "next/link";
import { PlusCircle } from "lucide-react";
import JobsTable from "@/components/JobsTable";

export default function JobsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Jobs</h1>
          <p className="text-[var(--muted)] mt-1">All training jobs and queue</p>
        </div>
        <Link
          href="/jobs/new"
          className="flex items-center gap-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
        >
          <PlusCircle size={15} />
          New Job
        </Link>
      </div>
      <JobsTable />
    </div>
  );
}
