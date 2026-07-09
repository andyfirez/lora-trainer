import JobQueuePanel from "@/components/JobQueuePanel";
import JobsList from "@/components/JobsList";
import PageHeader from "@/components/ui/PageHeader";
import { CardTitle } from "@/components/ui/Card";

export default function JobsPage() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Jobs"
        description="Training and sampling job runs"
      />

      <section className="space-y-4">
        <CardTitle>Queue</CardTitle>
        <JobQueuePanel />
      </section>

      <section className="space-y-4">
        <CardTitle>All Jobs</CardTitle>
        <JobsList />
      </section>
    </div>
  );
}
