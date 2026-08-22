"use client";

import { useState } from "react";
import { Copy } from "lucide-react";
import type { GalleryEntry } from "@/components/playground/galleryTypes";

interface PlaygroundGalleryProps {
  item: GalleryEntry | null;
}

export default function PlaygroundGallery({ item }: PlaygroundGalleryProps) {
  const [copied, setCopied] = useState(false);

  async function copyInfotext() {
    if (!item?.infotext) return;
    await navigator.clipboard.writeText(item.infotext);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
      <div className="flex min-h-0 flex-1 items-center justify-center p-3">
        {item ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={item.url}
            alt={item.kind}
            className="max-h-full max-w-full object-contain shadow-lg"
            onError={(event) => {
              event.currentTarget.style.visibility = "hidden";
            }}
            onLoad={(event) => {
              event.currentTarget.style.visibility = "visible";
            }}
          />
        ) : (
          <p className="text-sm text-muted">Generate to see preview</p>
        )}
      </div>
      {item?.infotext ? (
        <div className="border-t border-border px-3 py-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-muted">Infotext</span>
            <button
              type="button"
              onClick={() => void copyInfotext()}
              className="inline-flex items-center gap-1 text-xs text-muted hover:text-text"
            >
              <Copy size={12} />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="max-h-24 overflow-auto whitespace-pre-wrap text-[11px] leading-snug text-text-secondary">
            {item.infotext}
          </pre>
        </div>
      ) : null}
    </div>
  );
}
