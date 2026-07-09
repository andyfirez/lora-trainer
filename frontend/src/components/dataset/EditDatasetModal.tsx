"use client";

import { useEffect, useState } from "react";
import PathInput from "@/components/PathInput";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Button from "@/components/ui/Button";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { datasetsApi } from "@/lib/api/datasets";

interface Props {
  open: boolean;
  dataset: { id: number; name: string; image_dir: string };
  onClose: () => void;
  onSaved: () => void;
}

export default function EditDatasetModal({ open, dataset, onClose, onSaved }: Props) {
  const [name, setName] = useState(dataset.name);
  const [imageDir, setImageDir] = useState(dataset.image_dir);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setName(dataset.name);
      setImageDir(dataset.image_dir);
      setError(null);
    }
  }, [open, dataset.name, dataset.image_dir]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name || !imageDir) {
      setError("Name and image directory are required");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await datasetsApi.update(dataset.id, { name, image_dir: imageDir });
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
        <PathInput
          label="Image Directory"
          value={imageDir}
          onChange={setImageDir}
          placeholder="/path/to/images"
          pickerTitle="Select Image Directory"
          kind="directory"
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
