"use client";

import { useCallback, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/cn";

interface ResizeSplitProps {
  left: ReactNode;
  right: ReactNode;
  initialLeftPct?: number;
}

export default function ResizeSplit({ left, right, initialLeftPct = 200 / 3 }: ResizeSplitProps) {
  const [leftPct, setLeftPct] = useState(initialLeftPct);
  const dragging = useRef(false);
  const container = useRef<HTMLDivElement>(null);

  const onPointerMove = useCallback((event: PointerEvent) => {
    if (!dragging.current || !container.current) return;
    const rect = container.current.getBoundingClientRect();
    const next = ((event.clientX - rect.left) / rect.width) * 100;
    setLeftPct(Math.min(80, Math.max(28, next)));
  }, []);

  const stopDrag = useCallback(() => {
    dragging.current = false;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", stopDrag);
  }, [onPointerMove]);

  const startDrag = () => {
    dragging.current = true;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopDrag);
  };

  return (
    <div ref={container} className="flex min-h-0 flex-1">
      <div className="min-w-0 overflow-y-auto" style={{ width: `${leftPct}%` }}>
        {left}
      </div>
      <button
        type="button"
        aria-label="Resize panels"
        onPointerDown={startDrag}
        className={cn(
          "w-1.5 shrink-0 cursor-col-resize border-x border-border bg-border/60",
          "hover:bg-sampling/60 focus-visible:bg-sampling",
        )}
      />
      <div className="min-w-0 flex-1 overflow-hidden">{right}</div>
    </div>
  );
}
