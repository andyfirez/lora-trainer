"use client";

import { Play, Square } from "lucide-react";
import Button from "@/components/ui/Button";
import { canCancel, canEnqueue, cancelActionLabel } from "@/lib/runnableStatus";
import type { RunnableStatus } from "@/types";

interface RunnableEnqueueCancelButtonsProps {
  status: RunnableStatus;
  onEnqueue: () => void;
  onCancel: () => void;
}

export default function RunnableEnqueueCancelButtons({
  status,
  onEnqueue,
  onCancel,
}: RunnableEnqueueCancelButtonsProps) {
  return (
    <>
      {canEnqueue(status) && (
        <Button variant="success" size="sm" onClick={() => void onEnqueue()}>
          <Play size={13} /> Enqueue
        </Button>
      )}
      {canCancel(status) && (
        <Button variant="danger" size="sm" onClick={() => void onCancel()}>
          <Square size={13} /> {cancelActionLabel(status)}
        </Button>
      )}
    </>
  );
}
