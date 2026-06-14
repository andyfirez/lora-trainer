import JobQueuePanel from "@/components/JobQueuePanel";
import JobsList from "@/components/JobsList";

export default function JobsPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Jobs</h1>
        <p className="text-[var(--muted)] mt-1">Training and sampling job runs</p>
      </div>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Queue</h2>
        <JobQueuePanel />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">All Jobs</h2>
        <JobsList />
      </section>
    </div>
  );
}
