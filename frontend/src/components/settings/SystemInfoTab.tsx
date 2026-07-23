"use client";

import useSWR from "swr";
import Card from "@/components/ui/Card";
import { settingsApi } from "@/lib/api/settings";

function InfoRow({ label, value }: { label: string; value: string | number | boolean }) {
  const display = typeof value === "boolean" ? (value ? "Yes" : "No") : String(value);
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-1 py-2 border-b border-border last:border-b-0">
      <dt className="text-sm font-medium text-muted">{label}</dt>
      <dd className="text-sm text-text sm:col-span-2 break-all">{display}</dd>
    </div>
  );
}

export default function SystemInfoTab() {
  const { data, isLoading } = useSWR("/settings", () => settingsApi.get());

  if (isLoading || !data) {
    return <div className="text-muted">Loading…</div>;
  }

  const gpuVram =
    data.gpu.vram_gb && data.gpu.vram_gb.length > 0
      ? data.gpu.vram_gb.map((gb, index) => `GPU ${index}: ${gb} GB`).join(", ")
      : "—";

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <h3 className="text-sm font-semibold text-text mb-3">Server</h3>
        <dl>
          <InfoRow label="Host" value={data.server.host} />
          <InfoRow label="Port" value={data.server.port} />
        </dl>
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-text mb-3">Database</h3>
        <dl>
          <InfoRow label="Path" value={data.database.path} />
          <InfoRow label="SQL echo" value={data.database.echo} />
        </dl>
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-text mb-3">Training</h3>
        <dl>
          <InfoRow label="Logs directory" value={data.training.logs_dir} />
          <InfoRow label="Cancel poll interval (s)" value={data.training.cancel_poll_interval_seconds} />
        </dl>
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-text mb-3">Application</h3>
        <dl>
          <InfoRow label="Version" value={data.app_version} />
          <InfoRow label="Config file" value={data.config_file} />
        </dl>
      </Card>

      <Card className="lg:col-span-2">
        <h3 className="text-sm font-semibold text-text mb-3">GPU</h3>
        <dl>
          <InfoRow label="CUDA available" value={data.gpu.cuda_available} />
          <InfoRow label="Device name" value={data.gpu.device_name ?? "—"} />
          <InfoRow label="Device count" value={data.gpu.device_count} />
          <InfoRow label="VRAM" value={gpuVram} />
        </dl>
      </Card>
    </div>
  );
}
