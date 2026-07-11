"use client";

import { Loader2 } from "lucide-react";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Button from "@/components/ui/Button";

interface StopJobDialogProps {
  open: boolean;
  jobName: string;
  canSaveCheckpoint: boolean;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onStopNow: () => void;
  onSaveAndStop: () => void;
}

export default function StopJobDialog({
  open,
  jobName,
  canSaveCheckpoint,
  loading,
  error,
  onClose,
  onStopNow,
  onSaveAndStop,
}: StopJobDialogProps) {
  return (
    <Modal
      open={open}
      onClose={loading ? () => {} : onClose}
      title="Stop training job"
      description={
        canSaveCheckpoint
          ? `Choose how to stop "${jobName}".`
          : `Training has not started yet — no checkpoint can be saved. Stop "${jobName}" now?`
      }
      size="sm"
    >
      {error && <ModalError>{error}</ModalError>}
      <ModalFooter className="flex-col sm:flex-row sm:justify-end">
        <Button variant="secondary" size="sm" onClick={onClose} disabled={loading}>
          Don&apos;t stop
        </Button>
        <Button variant="danger" size="sm" onClick={onStopNow} disabled={loading}>
          {loading ? <Loader2 size={13} className="animate-spin" /> : null}
          Stop now
        </Button>
        {canSaveCheckpoint && (
          <Button variant="primary" size="sm" onClick={onSaveAndStop} disabled={loading}>
            {loading ? <Loader2 size={13} className="animate-spin" /> : null}
            Save checkpoint &amp; stop
          </Button>
        )}
      </ModalFooter>
    </Modal>
  );
}
