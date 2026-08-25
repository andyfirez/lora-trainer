"use client";

import { useEffect, useState } from "react";
import InferenceLoraSelect, {
  inferenceLoraNameForPath,
} from "@/components/lora/InferenceLoraSelect";
import Button from "@/components/ui/Button";
import Modal, { ModalFooter } from "@/components/ui/Modal";
import { inputClassName, labelClassName } from "@/components/ui/Input";
import { useInferenceLoras } from "@/hooks/useInferenceLoras";

interface AddLoraModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (name: string, weight: number) => void;
}

export default function AddLoraModal({ open, onClose, onConfirm }: AddLoraModalProps) {
  const { data } = useInferenceLoras();
  const [relativePath, setRelativePath] = useState("");
  const [weight, setWeight] = useState(1);

  useEffect(() => {
    if (!open) return;
    setRelativePath("");
    setWeight(1);
  }, [open]);

  const loras = data?.loras ?? [];
  const selectedName = inferenceLoraNameForPath(relativePath, loras);
  const canConfirm = Boolean(selectedName);

  return (
    <Modal open={open} onClose={onClose} title="Add LoRA" size="sm">
      <div className="space-y-4">
        <div className="space-y-1">
          <label className={labelClassName} htmlFor="add-lora-select">
            LoRA
          </label>
          <InferenceLoraSelect
            id="add-lora-select"
            value={relativePath}
            onChange={setRelativePath}
          />
        </div>
        <div className="space-y-1">
          <label className={labelClassName} htmlFor="add-lora-weight">
            Weight
          </label>
          <input
            id="add-lora-weight"
            type="number"
            min={0}
            max={2}
            step={0.1}
            className={inputClassName}
            value={weight}
            onChange={(event) =>
              setWeight(event.target.value === "" ? 1 : Number(event.target.value))
            }
          />
        </div>
      </div>
      <ModalFooter>
        <Button type="button" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button
          type="button"
          disabled={!canConfirm}
          onClick={() => {
            if (!selectedName) return;
            onConfirm(selectedName, weight);
            onClose();
          }}
        >
          OK
        </Button>
      </ModalFooter>
    </Modal>
  );
}
