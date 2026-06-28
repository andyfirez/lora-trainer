"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Cropper, {
  type Area,
  type MediaSize,
  getInitialCropFromCroppedAreaPixels,
} from "react-easy-crop";
import { X } from "lucide-react";
import { datasetCropPreviewUrl, datasetsApi } from "@/lib/api/datasets";
import type { CropMeta } from "@/types";

interface Props {
  datasetId: number;
  filename: string;
  targetResolution: number;
  onClose: () => void;
  onSaved: () => void;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 3;
const PREVIEW_SIZE = 160;

function centerToArea(meta: CropMeta, resolution: number): Area {
  const half = resolution / 2;
  const x = Math.max(0, Math.min(meta.crop_center_x * meta.fitted_width - half, meta.fitted_width - resolution));
  const y = Math.max(0, Math.min(meta.crop_center_y * meta.fitted_height - half, meta.fitted_height - resolution));
  return { x, y, width: resolution, height: resolution };
}

function areaToCenter(area: Area, meta: CropMeta): { x: number; y: number } {
  const half = area.width / 2;
  const cx = Math.max(half, Math.min(area.x + half, meta.fitted_width - half));
  const cy = Math.max(half, Math.min(area.y + half, meta.fitted_height - half));
  return {
    x: cx / meta.fitted_width,
    y: cy / meta.fitted_height,
  };
}

function computeDisplayCropSize(mediaSize: MediaSize, targetResolution: number): { width: number; height: number } {
  const scale = mediaSize.width / mediaSize.naturalWidth;
  const size = Math.round(targetResolution * scale);
  const maxSize = Math.floor(Math.min(mediaSize.width, mediaSize.height));
  const clamped = Math.min(size, maxSize);
  return { width: clamped, height: clamped };
}

interface CropperSession {
  mediaSize: MediaSize;
  cropSize: { width: number; height: number };
  crop: { x: number; y: number };
  zoom: number;
  initialArea: Area;
}

function drawCropPreview(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  area: Area,
  outputSize: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  canvas.width = outputSize;
  canvas.height = outputSize;
  ctx.clearRect(0, 0, outputSize, outputSize);
  ctx.drawImage(image, area.x, area.y, area.width, area.height, 0, 0, outputSize, outputSize);
}

export default function ImageCropModal({ datasetId, filename, targetResolution, onClose, onSaved }: Props) {
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const sourceImageRef = useRef<HTMLImageElement | null>(null);
  const [meta, setMeta] = useState<CropMeta | null>(null);
  const [session, setSession] = useState<CropperSession | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area | null>(null);
  const [sourceImageReady, setSourceImageReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageUrl = useMemo(() => datasetCropPreviewUrl(datasetId, filename), [datasetId, filename]);

  useEffect(() => {
    let cancelled = false;
    setSourceImageReady(false);
    sourceImageRef.current = null;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      sourceImageRef.current = img;
      setSourceImageReady(true);
    };
    img.onerror = () => {
      if (!cancelled) setError("Failed to load image preview");
    };
    img.src = imageUrl;
    return () => {
      cancelled = true;
    };
  }, [imageUrl]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      setSession(null);
      try {
        const loaded = await datasetsApi.getCropMeta(datasetId, filename);
        if (cancelled) return;
        setMeta(loaded);
        setCroppedAreaPixels(centerToArea(loaded, targetResolution));
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load crop");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [datasetId, filename, targetResolution]);

  const updatePreview = useCallback((area: Area | null) => {
    const canvas = previewCanvasRef.current;
    const image = sourceImageRef.current;
    if (!canvas || !image || !area) return;
    drawCropPreview(canvas, image, area, PREVIEW_SIZE);
  }, []);

  useEffect(() => {
    updatePreview(croppedAreaPixels);
  }, [croppedAreaPixels, sourceImageReady, updatePreview]);

  const handleMediaLoaded = useCallback(
    (mediaSize: MediaSize) => {
      if (!meta || session) return;
      const displayCropSize = computeDisplayCropSize(mediaSize, targetResolution);
      const initialArea = centerToArea(meta, targetResolution);
      const initial = getInitialCropFromCroppedAreaPixels(
        initialArea,
        mediaSize,
        0,
        displayCropSize,
        MIN_ZOOM,
        MAX_ZOOM
      );
      setSession({
        mediaSize,
        cropSize: displayCropSize,
        crop: initial.crop,
        zoom: initial.zoom,
        initialArea,
      });
      setCrop(initial.crop);
      setZoom(initial.zoom);
      setCroppedAreaPixels(initialArea);
    },
    [meta, session, targetResolution]
  );

  const handleCropAreaChange = useCallback((_cropped: Area, pixels: Area) => {
    setCroppedAreaPixels(pixels);
  }, []);

  const handleSave = async () => {
    if (!meta || !croppedAreaPixels) return;
    setSaving(true);
    setError(null);
    try {
      const normalizedArea: Area = {
        x: croppedAreaPixels.x,
        y: croppedAreaPixels.y,
        width: targetResolution,
        height: targetResolution,
      };
      const center = areaToCenter(normalizedArea, meta);
      await datasetsApi.saveCrop(datasetId, filename, center.x, center.y);
      await datasetsApi.bakeImage(datasetId, filename);
      onSaved();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl w-full max-w-4xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <div>
            <div className="text-sm font-medium text-white">Crop image</div>
            <div className="text-xs text-[var(--muted)] truncate max-w-md">{filename}</div>
          </div>
          <button type="button" onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-[var(--muted)]">
            <X size={18} />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-0">
          <div className="relative h-[min(60vh,520px)] bg-neutral-900">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center text-[var(--muted)] text-sm">
                Loading…
              </div>
            ) : meta && session ? (
              <Cropper
                image={imageUrl}
                crop={crop}
                zoom={zoom}
                aspect={1}
                minZoom={MIN_ZOOM}
                maxZoom={MAX_ZOOM}
                cropSize={session.cropSize}
                objectFit="contain"
                onCropChange={setCrop}
                onZoomChange={setZoom}
                onCropAreaChange={handleCropAreaChange}
                restrictPosition
              />
            ) : meta ? (
              <Cropper
                image={imageUrl}
                crop={{ x: 0, y: 0 }}
                zoom={MIN_ZOOM}
                aspect={1}
                onCropChange={() => {}}
                onMediaLoaded={handleMediaLoaded}
                restrictPosition
                objectFit="contain"
                style={{ containerStyle: { visibility: "hidden" } }}
              />
            ) : null}
            {!loading && meta && !session && (
              <div className="absolute inset-0 flex items-center justify-center text-[var(--muted)] text-sm">
                Preparing crop…
              </div>
            )}
          </div>

          <div className="border-t md:border-t-0 md:border-l border-[var(--border)] p-4 flex md:flex-col items-center justify-center gap-2 bg-[var(--bg)]">
            <div className="text-xs text-[var(--muted)]">Result preview</div>
            <canvas
              ref={previewCanvasRef}
              className="rounded-lg border border-[var(--border)] bg-black"
              style={{ width: PREVIEW_SIZE, height: PREVIEW_SIZE }}
            />
            <div className="text-[10px] text-[var(--muted)] text-center max-w-[160px]">
              Updates as you move the image under the crop frame
            </div>
          </div>
        </div>

        <div className="px-4 py-3 space-y-3 border-t border-[var(--border)]">
          <div className="flex items-center gap-3">
            <label className="text-xs text-[var(--muted)] shrink-0">Zoom</label>
            <input
              type="range"
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={0.05}
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
              className="flex-1"
              disabled={!session}
            />
          </div>
          {error && <div className="text-sm text-red-400">{error}</div>}
          <div className="flex gap-2 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm border border-[var(--border)] rounded-lg text-[var(--muted)] hover:text-white"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || !session}
              className="px-4 py-2 text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save & bake"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
