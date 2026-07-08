"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { PlusCircle, Loader2, Trash2, Copy } from "lucide-react";
import { configsApi } from "@/lib/api/configs";
import type { ConfigType } from "@/types";

export default function ConfigsPage() {
  const router = useRouter();
  const [tab, setTab] = useState<ConfigType>("training");
  const [cloningId, setCloningId] = useState<number | null>(null);
  const { data: configs, isLoading, mutate } = useSWR(
    `/configs?type=${tab}`,
    () => configsApi.list(tab),
    { refreshInterval: 10000 },
  );

  const handleDelete = async (id: number, name: string) => {
    if (!confirm(`Delete config "${name}"?`)) return;
    await configsApi.delete(id);
    mutate();
  };

  const handleClone = async (id: number) => {
    setCloningId(id);
    try {
      const cloned = await configsApi.clone(id);
      await mutate();
      router.push(`/configs/${cloned.id}`);
    } finally {
      setCloningId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Configs</h1>
          <p className="text-[var(--muted)] mt-1">Reusable training and sampling configurations</p>
        </div>
        <Link
          href={`/configs/new?type=${tab}`}
          className="flex items-center gap-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
        >
          <PlusCircle size={15} />
          New Config
        </Link>
      </div>

      <div className="flex gap-1 border-b border-[var(--border)]">
        {(["training", "sampling"] as ConfigType[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors capitalize ${
              tab === t
                ? "text-white border border-b-[var(--bg)] border-[var(--border)] bg-[var(--bg)] -mb-px"
                : "text-[var(--muted)] hover:text-white"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20 text-[var(--muted)]">
          <Loader2 className="animate-spin mr-2" size={18} /> Loading configs…
        </div>
      ) : !configs?.length ? (
        <div className="text-center py-20 text-[var(--muted)] rounded-xl border border-[var(--border)]">
          No {tab} configs yet.{" "}
          <Link href={`/configs/new?type=${tab}`} className="text-[var(--accent)] hover:underline">
            Create one
          </Link>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
          <table className="w-full text-sm">
            <thead className="bg-[var(--surface)]">
              <tr>
                <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Name</th>
                {tab === "training" && (
                  <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Version</th>
                )}
                <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Description</th>
                <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Updated</th>
                <th className="px-4 py-3 text-right text-[var(--muted)] font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {configs.map((config) => (
                <tr key={config.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-4 py-3">
                    <Link
                      href={`/configs/${config.id}`}
                      className="text-white hover:text-[var(--accent)] font-medium"
                    >
                      {config.name}
                    </Link>
                  </td>
                  {tab === "training" && (
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center rounded-full bg-[var(--accent)]/15 text-[var(--accent)] px-2.5 py-0.5 text-xs font-medium">
                        v{config.active_version ?? 1}
                      </span>
                    </td>
                  )}
                  <td className="px-4 py-3 text-[var(--muted)] max-w-xs truncate">
                    {config.description || "—"}
                  </td>
                  <td className="px-4 py-3 text-[var(--muted)]">
                    {new Date(config.updated_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => void handleClone(config.id)}
                        disabled={cloningId === config.id}
                        title="Duplicate"
                        className="p-1.5 rounded hover:bg-white/10 text-[var(--muted)] hover:text-white disabled:opacity-50"
                      >
                        {cloningId === config.id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Copy size={14} />
                        )}
                      </button>
                      <button
                        onClick={() => void handleDelete(config.id, config.name)}
                        title="Delete"
                        className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
