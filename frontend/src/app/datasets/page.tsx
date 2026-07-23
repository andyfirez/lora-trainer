"use client";

import useSWR from "swr";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useState } from "react";
import { PlusCircle } from "lucide-react";
import PathInput from "@/components/PathInput";
import StorageFolderBrowser from "@/components/storage/StorageFolderBrowser";
import DatasetFolderItem from "@/components/dataset/DatasetFolderItem";
import { datasetsApi } from "@/lib/api/datasets";
import PageHeader from "@/components/ui/PageHeader";
import Button from "@/components/ui/Button";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import { joinRelativePath, normalizeRelativePath } from "@/lib/storagePaths";

function ImportDatasetModal({
  parentPath,
  onClose,
  onCreated,
}: {
  parentPath: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [sourceDir, setSourceDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const importPath = name ? joinRelativePath(parentPath, name) : "";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !sourceDir) {
      setError("Fill in all required fields");
      return;
    }
    setSaving(true);
    try {
      await datasetsApi.import({
        name,
        source_dir: sourceDir,
        relative_path: joinRelativePath(parentPath, name),
      });
      onCreated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error importing dataset");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal open title="Import Dataset" onClose={onClose}>
      {error && <ModalError>{error}</ModalError>}
      <form onSubmit={handleSubmit} className="space-y-3">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-dataset"
        />
        {importPath ? (
          <p className="text-xs text-muted">Will import to: {importPath}</p>
        ) : (
          <p className="text-xs text-muted">Enter a name to see the import destination.</p>
        )}
        <PathInput
          label="Source folder (external)"
          value={sourceDir}
          onChange={setSourceDir}
          placeholder="/path/to/source"
          pickerTitle="Select folder to import"
          kind="directory"
        />
        <ModalFooter>
          <Button variant="secondary" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button type="submit" disabled={saving} className="flex-1">
            {saving ? "Importing…" : "Import"}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

function DatasetsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentPath = normalizeRelativePath(searchParams.get("path") ?? "");
  const { data: datasets, isLoading, mutate } = useSWR("/datasets", () => datasetsApi.list());
  const [showImport, setShowImport] = useState(false);

  const navigateToPath = useCallback(
    (path: string, replace = false) => {
      const normalized = normalizeRelativePath(path);
      const params = new URLSearchParams(searchParams.toString());
      if (normalized) {
        params.set("path", normalized);
      } else {
        params.delete("path");
      }
      const query = params.toString();
      const href = query ? `/datasets?${query}` : "/datasets";
      if (replace) {
        router.replace(href);
      } else {
        router.push(href);
      }
    },
    [router, searchParams]
  );

  const handleNavigate = useCallback(
    (path: string) => {
      const normalized = normalizeRelativePath(path);
      const current = normalizeRelativePath(currentPath);
      const isBack =
        normalized === "" ||
        (current.startsWith(`${normalized}/`) && normalized.split("/").length < current.split("/").length);
      navigateToPath(path, isBack);
    },
    [currentPath, navigateToPath]
  );

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
          <Button onClick={() => setShowImport(true)}>
            <PlusCircle size={15} />
            Import Dataset
          </Button>
        }
      />

      <StorageFolderBrowser
        kind="datasets"
        items={datasets ?? []}
        currentPath={currentPath}
        onNavigate={handleNavigate}
        catalogLoading={isLoading}
        emptyHint="Import a dataset or place images under the datasets root to auto-discover."
        renderItem={(dataset) => (
          <DatasetFolderItem dataset={dataset} onDelete={() => void handleDelete(dataset.id, dataset.name)} />
        )}
      />

      {showImport && (
        <ImportDatasetModal
          parentPath={currentPath}
          onClose={() => setShowImport(false)}
          onCreated={() => mutate()}
        />
      )}
    </div>
  );
}

export default function DatasetsPage() {
  return (
    <Suspense fallback={<div className="text-muted py-20">Loading…</div>}>
      <DatasetsPageContent />
    </Suspense>
  );
}
