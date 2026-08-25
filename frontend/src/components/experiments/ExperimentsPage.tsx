"use client";

import Link from "next/link";
import { PlusCircle } from "lucide-react";
import ExperimentListItem from "@/components/experiments/ExperimentListItem";
import PlaygroundQueue from "@/components/playground/PlaygroundQueue";
import PageHeader from "@/components/ui/PageHeader";
import { useExperiments } from "@/hooks/useExperiments";

export default function ExperimentsPage() {
  const { inProgress, completed, queueItems, error, cancelQueueItem } = useExperiments();

  return (
    <div className="-m-4 flex h-[100dvh] flex-col overflow-hidden bg-bg md:-m-6 lg:-m-8">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-4 py-6">
          <PageHeader
            title="Experiments"
            description="Parameter grids with history, queue, and results"
            actions={
              <Link
                href="/experiments/new"
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
              >
                <PlusCircle size={15} />
                New Experiment
              </Link>
            }
          />

          {error && <p className="text-sm text-error">{error}</p>}

          <section className="space-y-3">
            <h2 className="text-sm font-medium text-muted">In progress</h2>
            {inProgress.length === 0 ? (
              <p className="text-sm text-muted">No experiments running or queued.</p>
            ) : (
              <div className="space-y-2">
                {inProgress.map((experiment) => (
                  <ExperimentListItem key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </section>

          <section className="space-y-3">
            <h2 className="text-sm font-medium text-muted">Completed</h2>
            {completed.length === 0 ? (
              <p className="text-sm text-muted">No completed experiments yet.</p>
            ) : (
              <div className="space-y-2">
                {completed.map((experiment) => (
                  <ExperimentListItem key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
      <PlaygroundQueue items={queueItems} onCancel={(item) => void cancelQueueItem(item)} />
    </div>
  );
}
