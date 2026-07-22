"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { PlusCircle, Loader2, Trash2, Copy } from "lucide-react";
import { configsApi } from "@/lib/api/configs";
import PageHeader from "@/components/ui/PageHeader";
import Button from "@/components/ui/Button";
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from "@/components/ui/Table";
import Card from "@/components/ui/Card";

export default function ConfigsPage() {
  const router = useRouter();
  const [cloningId, setCloningId] = useState<number | null>(null);
  const { data: configs, isLoading, mutate } = useSWR("/configs/training", () => configsApi.list("training"), {
    refreshInterval: 10000,
  });

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
      <PageHeader
        title="Configs"
        description="Reusable SDXL LoRA training configurations"
        actions={
          <Link
            href="/configs/new"
            className="inline-flex items-center gap-2 bg-accent hover:bg-accent-hover text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            <PlusCircle size={15} />
            New Config
          </Link>
        }
      />

      {isLoading ? (
        <div className="flex items-center justify-center py-20 text-muted">
          <Loader2 className="animate-spin mr-2" size={18} /> Loading configs…
        </div>
      ) : !configs?.length ? (
        <Card className="text-center py-20 text-muted">
          No training configs yet.{" "}
          <Link href="/configs/new" className="text-accent hover:underline">
            Create one
          </Link>
        </Card>
      ) : (
        <Table>
          <TableHead>
            <tr>
              <TableHeader>Name</TableHeader>
              <TableHeader>Description</TableHeader>
              <TableHeader>Updated</TableHeader>
              <TableHeader className="text-right">Actions</TableHeader>
            </tr>
          </TableHead>
          <TableBody>
            {configs.map((config) => (
              <TableRow key={config.id}>
                <TableCell>
                  <Link href={`/configs/${config.id}`} className="text-text hover:text-accent font-medium">
                    {config.name}
                  </Link>
                </TableCell>
                <TableCell className="text-muted max-w-xs truncate">{config.description || "—"}</TableCell>
                <TableCell className="text-muted">{new Date(config.updated_at).toLocaleDateString()}</TableCell>
                <TableCell>
                  <div className="flex items-center justify-end gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => void handleClone(config.id)}
                      disabled={cloningId === config.id}
                      title="Duplicate"
                      aria-label="Duplicate config"
                    >
                      {cloningId === config.id ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Copy size={14} />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => void handleDelete(config.id, config.name)}
                      title="Delete"
                      aria-label="Delete config"
                      className="text-error hover:text-error"
                    >
                      <Trash2 size={14} />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
