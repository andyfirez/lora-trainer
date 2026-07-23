"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import Input from "@/components/ui/Input";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { settingsApi } from "@/lib/api/settings";

export default function WorkerSettingsTab() {
  const { data, isLoading, mutate } = useSWR("/settings", () => settingsApi.get());
  const [maxConcurrentJobs, setMaxConcurrentJobs] = useState("");
  const [pollInterval, setPollInterval] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (!data) return;
    setMaxConcurrentJobs(String(data.max_concurrent_jobs));
    setPollInterval(String(data.worker_poll_interval_seconds));
  }, [data]);

  const handleSave = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(false);

    const maxConcurrent = Number.parseInt(maxConcurrentJobs, 10);
    const pollSeconds = Number.parseInt(pollInterval, 10);
    if (!Number.isFinite(maxConcurrent) || maxConcurrent < 1) {
      setError("Max concurrent jobs must be an integer ≥ 1");
      return;
    }
    if (!Number.isFinite(pollSeconds) || pollSeconds < 1) {
      setError("Poll interval must be an integer ≥ 1");
      return;
    }

    setSaving(true);
    try {
      await settingsApi.patch({
        max_concurrent_jobs: maxConcurrent,
        worker_poll_interval_seconds: pollSeconds,
      });
      await mutate();
      setSuccess(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (isLoading || !data) {
    return <div className="text-muted">Loading…</div>;
  }

  return (
    <Card className="max-w-xl space-y-4">
      <p className="text-sm text-muted">
        Worker queue settings are saved to config.toml and applied without restarting the server.
      </p>
      <form onSubmit={handleSave} className="space-y-4">
        <Input
          label="Max concurrent jobs"
          type="number"
          min={1}
          value={maxConcurrentJobs}
          onChange={(e) => setMaxConcurrentJobs(e.target.value)}
        />
        <Input
          label="Worker poll interval (seconds)"
          type="number"
          min={1}
          value={pollInterval}
          onChange={(e) => setPollInterval(e.target.value)}
        />
        {error && <p className="text-sm text-error">{error}</p>}
        {success && <p className="text-sm text-success">Settings saved.</p>}
        <Button type="submit" disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </form>
    </Card>
  );
}
