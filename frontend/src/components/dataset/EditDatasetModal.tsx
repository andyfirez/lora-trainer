"use client";

import { useEffect, useState } from "react";
import StoragePathInput from "@/components/StoragePathInput";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Button from "@/components/ui/Button";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { datasetsApi } from "@/lib/api/datasets";

interface Props {
  open: boolean;
  dataset: { id: number; name: string; relative_path: string; path_missing?: boolean };
  onClose: () => void;
  onSaved: () => void;
}

export default function EditDatasetModal({ open, dataset, onClose, onSaved }: Props) {
  const [name, setName] = useState(dataset.name);
  const [relativePath, setRelativePath] = useState(dataset.relative_path);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setName(dataset.name);
      setRelativePath(dataset.relative_path);
      setError(null);
    }
  }, [open, dataset.name, dataset.relative_path]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name || !relativePath) {
      setError("Name and path are required");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await datasetsApi.update(dataset.id, { name, relative_path: relativePath });
      onSaved();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error updating dataset");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Edit Dataset">
      {error && <ModalError>{error}</ModalError>}
      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className={labelClassName}>Name</label>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="my-dataset"
            className={inputClassName}
          />
        </div>
        <StoragePathInput
          label="Path inside datasets root"
          value={relativePath}
          onChange={setRelativePath}
          kind="datasets"
          placeholder="anime/girl_01"
          warning={dataset.path_missing ? "Folder not found on disk" : null}
        />
        <ModalFooter>
          <Button type="button" variant="secondary" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button type="submit" disabled={saving} className="flex-1">
            {saving ? "Saving…" : "Save Changes"}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
