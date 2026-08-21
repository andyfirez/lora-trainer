"use client";

import { useRef } from "react";
import { Copy, ImageUp, Loader2 } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Alert from "@/components/ui/Alert";
import { usePngInspect } from "@/hooks/usePngInspect";
import { cn } from "@/lib/cn";

function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-1 py-2 border-b border-border last:border-b-0">
      <dt className="text-sm font-medium text-muted">{label}</dt>
      <dd className="text-sm text-text sm:col-span-2 break-words whitespace-pre-wrap">{String(value)}</dd>
    </div>
  );
}

const PARSED_PARAM_ORDER = [
  "Prompt",
  "Negative prompt",
  "Steps",
  "Sampler",
  "Schedule type",
  "CFG scale",
  "Seed",
  "Size-1",
  "Size-2",
  "Model hash",
  "Model",
  "Version",
];

function orderedParameterEntries(parameters: Record<string, string | number>) {
  const seen = new Set<string>();
  const entries: [string, string | number][] = [];

  for (const key of PARSED_PARAM_ORDER) {
    if (key in parameters) {
      entries.push([key, parameters[key]]);
      seen.add(key);
    }
  }

  for (const [key, value] of Object.entries(parameters)) {
    if (!seen.has(key)) {
      entries.push([key, value]);
    }
  }

  return entries;
}

function orderedMetadataEntries(items: Record<string, string>) {
  const entries = Object.entries(items);
  entries.sort(([a], [b]) => {
    if (a === "parameters") return -1;
    if (b === "parameters") return 1;
    return a.localeCompare(b);
  });
  return entries;
}

export default function PngInfoPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const {
    dragActive,
    loading,
    error,
    fileName,
    localPreviewUrl,
    result,
    copied,
    handleFiles,
    setDragActive,
    copyRawInfo,
  } = usePngInspect();

  const previewSrc = result?.preview_base64 ?? localPreviewUrl;
  const metadataEntries = result ? orderedMetadataEntries(result.items) : [];
  const parameterEntries = result ? orderedParameterEntries(result.parameters) : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <PageHeader
        title="PNG Info"
        description="Inspect generation metadata embedded in images"
        actions={
          result?.info ? (
            <Button variant="secondary" size="sm" onClick={() => void copyRawInfo()}>
              <Copy size={14} />
              {copied ? "Copied" : "Copy raw info"}
            </Button>
          ) : undefined
        }
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr]">
        <Card className="space-y-4">
          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              setDragActive(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              handleFiles(event.dataTransfer.files);
            }}
            className={cn(
              "relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-8 text-center transition-colors",
              dragActive
                ? "border-accent bg-accent/10"
                : "border-border bg-white/[0.02] hover:border-accent/60 hover:bg-white/[0.04]",
            )}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => handleFiles(event.target.files)}
            />
            {loading ? (
              <Loader2 size={28} className="animate-spin text-accent" />
            ) : (
              <ImageUp size={28} className="text-muted" />
            )}
            <p className="mt-3 text-sm font-medium text-text">
              {loading ? "Reading metadata…" : "Drop an image here or click to upload"}
            </p>
            <p className="mt-1 text-xs text-muted">PNG, JPEG, WebP, GIF</p>
          </div>

          {previewSrc && (
            <div className="space-y-2">
              {fileName && <p className="truncate text-xs text-muted">{fileName}</p>}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewSrc}
                alt={fileName ?? "Uploaded preview"}
                className="max-h-[420px] w-full rounded-lg border border-border object-contain bg-black/20"
              />
              {result && (
                <p className="text-xs text-muted">
                  {result.width} × {result.height}px
                </p>
              )}
            </div>
          )}
        </Card>

        <div className="space-y-4">
          {error && <Alert variant="error">{error}</Alert>}

          {!loading && result && metadataEntries.length === 0 && (
            <Card>
              <p className="text-sm text-muted">Nothing found in the image.</p>
            </Card>
          )}

          {metadataEntries.map(([key, value]) => (
            <Card
              key={key}
              className={cn(key === "parameters" && "border-accent/40 bg-accent/5")}
            >
              <h3 className="mb-2 text-sm font-semibold text-text">{key}</h3>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-text-secondary">
                {value}
              </pre>
            </Card>
          ))}

          {parameterEntries.length > 0 && (
            <Card>
              <h3 className="mb-3 text-sm font-semibold text-text">Parsed parameters</h3>
              <dl>
                {parameterEntries.map(([key, value]) => (
                  <InfoRow key={key} label={key} value={value} />
                ))}
              </dl>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
