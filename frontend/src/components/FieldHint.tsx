"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { CircleHelp } from "lucide-react";
import { parameterAnchor } from "@/lib/parameterUtils";

interface FieldHintProps {
  hint: string;
  hintAnchor?: string;
}

export default function FieldHint({ hint, hintAnchor }: FieldHintProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  const anchor = hintAnchor ? parameterAnchor(hintAnchor) : undefined;

  return (
    <div
      ref={containerRef}
      className="relative inline-flex align-middle ml-1"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="inline-flex items-center justify-center text-[var(--muted)] hover:text-white transition-colors"
        aria-label="Parameter help"
        onClick={() => setOpen((prev) => !prev)}
      >
        <CircleHelp size={13} />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-50 pt-1">
          <div className="w-64 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 shadow-lg text-xs text-[var(--muted)] leading-relaxed">
            <p>{hint}</p>
            {anchor && (
              <Link
                href={`/parameters#${anchor}`}
                className="mt-2 inline-block text-[var(--accent)] hover:underline"
                onClick={() => setOpen(false)}
              >
                Learn more →
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
