"use client";

import useSWR from "swr";
import Link from "next/link";
import { useState } from "react";
import { PlusCircle, Trash2, Image as ImageIcon } from "lucide-react";
import PathInput from "@/components/PathInput";
import StoragePathInput from "@/components/StoragePathInput";
import { datasetsApi } from "@/lib/api/datasets";
import type { Dataset } from "@/types";
import PageHeader from "@/components/ui/PageHeader";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Badge from "@/components/ui/Badge";
import Tabs from "@/components/ui/Tabs";

function CreateDatasetModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [mode, setMode] = useState<"register" | "import">("register");
  const [name, setName] = useState("");
  const [relativePath, setRelativePath] = useState("");
  const [sourceDir, setSourceDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !relativePath || (mode === "import" && !sourceDir)) {
      setError("Fill in all required fields");
      return;
    }
    setSaving(true);
    try {
      if (mode === "import") {
        await datasetsApi.import({ name, source_dir: sourceDir, relative_path: relativePath });
      } else {
        await datasetsApi.create({ name, relative_path: relativePath });
      }
      onCreated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error creating dataset");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal open title="Add Dataset" onClose={onClose}>
      {error && <ModalError>{error}</ModalError>}
      <form onSubmit={handleSubmit} className="space-y-3">
        <Tabs
          tabs={[
            { value: "register", label: "Existing folder" },
            { value: "import", label: "Import & copy" },
          ]}
          value={mode}
          onChange={setMode}
        />
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-dataset"
        />
        <StoragePathInput
          label="Path inside datasets root"
          value={relativePath}
          onChange={setRelativePath}
          kind="datasets"
          placeholder="anime/girl_01"
          hint={mode === "register" ? "Folder must already exist under datasets root" : "Destination path for copied files"}
        />
        {mode === "import" && (
          <PathInput
            label="Source folder (external)"
            value={sourceDir}
            onChange={setSourceDir}
            placeholder="/path/to/source"
            pickerTitle="Select folder to import"
            kind="directory"
          />
        )}
        <ModalFooter>
          <Button variant="secondary" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button type="submit" disabled={saving} className="flex-1">
            {saving ? "Adding…" : mode === "import" ? "Import" : "Add Dataset"}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

function DatasetCard({ dataset, onDelete }: { dataset: Dataset; onDelete: () => void }) {
  const { data: images } = useSWR(`/datasets/${dataset.id}/images`, () => datasetsApi.listImages(dataset.id));

  return (
    <Link
      href={`/datasets/${dataset.id}`}
      className="block rounded-xl border border-border bg-surface p-5 space-y-3 hover:border-accent transition-colors shadow-sm"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="font-semibold text-text">{dataset.name}</div>
          <div className="text-xs text-muted mt-0.5 break-all">{dataset.relative_path}</div>
          <div className="flex flex-wrap gap-2 mt-2">
            {dataset.target_resolution != null && (
              <Badge>{dataset.target_resolution}px</Badge>
            )}
            <Badge variant={dataset.preprocess_ready ? "success" : "warning"}>
              {dataset.preprocess_ready ? "Ready" : "Not prepared"}
            </Badge>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={(event) => {
            event.preventDefault();
            onDelete();
          }}
          className="text-error hover:text-error shrink-0"
          aria-label="Delete dataset"
        >
          <Trash2 size={14} />
        </Button>
      </div>
      <div className="flex items-center gap-2 text-sm text-muted">
        <ImageIcon size={14} />
        <span>{images?.images.length ?? "…"} images</span>
      </div>
    </Link>
  );
}

export default function DatasetsPage() {
  const { data: datasets, isLoading, mutate } = useSWR("/datasets", () => datasetsApi.list());
  const [showCreate, setShowCreate] = useState(false);

  const handleDelete = async (id: number, name: string) => {
    if (!confirm(`Delete dataset "${name}"?`)) return;
    await datasetsApi.delete(id);
    mutate();
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Datasets"
        description="Manage your training image datasets and tags"
        actions={
          <Button onClick={() => setShowCreate(true)}>
            <PlusCircle size={15} />
            Add Dataset
          </Button>
        }
      />

      {isLoading ? (
        <div className="text-muted">Loading…</div>
      ) : !datasets?.length ? (
        <Card className="text-center py-20 text-muted">
          No datasets yet. Add one or place images under the datasets root to auto-discover.
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {datasets.map((d) => (
            <DatasetCard key={d.id} dataset={d} onDelete={() => handleDelete(d.id, d.name)} />
          ))}
        </div>
      )}

      {showCreate && <CreateDatasetModal onClose={() => setShowCreate(false)} onCreated={() => mutate()} />}
    </div>
  );
}
