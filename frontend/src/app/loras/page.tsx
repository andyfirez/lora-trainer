"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2 } from "lucide-react";
import { lorasApi } from "@/lib/api/loras";
import PageHeader from "@/components/ui/PageHeader";
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from "@/components/ui/Table";
import Card from "@/components/ui/Card";

export default function LorasPage() {
  const { data: loras, isLoading } = useSWR("/loras", () => lorasApi.list());

  return (
    <div className="space-y-6">
      <PageHeader
        title="LoRAs"
        description="Successfully trained LoRA models with frozen configs and artifacts"
      />

      {isLoading ? (
        <div className="flex items-center justify-center py-20 text-muted">
          <Loader2 className="animate-spin mr-2" size={18} /> Loading LoRAs…
        </div>
      ) : !loras?.length ? (
        <Card className="text-center py-20 text-muted">
          No trained LoRAs yet. Complete a training job to see results here.
        </Card>
      ) : (
        <Table>
          <TableHead>
            <tr>
              <TableHeader>Name</TableHeader>
              <TableHeader>Base Model</TableHeader>
              <TableHeader>Job</TableHeader>
              <TableHeader>Created</TableHeader>
            </tr>
          </TableHead>
          <TableBody>
            {loras.map((lora) => (
              <TableRow key={lora.id}>
                <TableCell>
                  <Link href={`/loras/${lora.id}`} className="text-text hover:text-accent font-medium">
                    {lora.name}
                  </Link>
                </TableCell>
                <TableCell className="text-muted max-w-xs truncate">{lora.base_model_name}</TableCell>
                <TableCell>
                  <Link href={`/jobs/${lora.job_id}`} className="text-accent hover:underline text-sm">
                    Job #{lora.job_id}
                  </Link>
                </TableCell>
                <TableCell className="text-muted">{new Date(lora.created_at).toLocaleDateString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
