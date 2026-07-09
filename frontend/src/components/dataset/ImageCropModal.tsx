"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Cropper, {
  type Area,
  type MediaSize,
  getInitialCropFromCroppedAreaPixels,
} from "react-easy-crop";
import { X } from "lucide-react";
import { Modal, ModalFooter, Button } from "@/components/ui";
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
const PREVIEW_MAX = 160;

function cropDimensions(meta: CropMeta, targetResolution: number): { width: number; height: number } {
  if (meta.enable_bucket && meta.bucket_width && meta.bucket_height) {
    return { width: meta.bucket_width, height: meta.bucket_height };
  }
  return { width: targetResolution, height: targetResolution };
}

function centerToArea(meta: CropMeta, cropWidth: number, cropHeight: number): Area {
  const halfW = cropWidth / 2;
  const halfH = cropHeight / 2;
  const x = Math.max(0, Math.min(meta.crop_center_x * meta.fitted_width - halfW, meta.fitted_width - cropWidth));
  const y = Math.max(0, Math.min(meta.crop_center_y * meta.fitted_height - halfH, meta.fitted_height - cropHeight));
  return { x, y, width: cropWidth, height: cropHeight };
}

function areaToCenter(area: Area, meta: CropMeta): { x: number; y: number } {
  const halfW = area.width / 2;
  const halfH = area.height / 2;
  const cx = Math.max(halfW, Math.min(area.x + halfW, meta.fitted_width - halfW));
  const cy = Math.max(halfH, Math.min(area.y + halfH, meta.fitted_height - halfH));
  return {
    x: cx / meta.fitted_width,
    y: cy / meta.fitted_height,
  };
}

function computeDisplayCropSize(
  mediaSize: MediaSize,
  cropWidth: number,
  cropHeight: number
): { width: number; height: number } {
  const scale = mediaSize.width / mediaSize.naturalWidth;
  const width = Math.round(cropWidth * scale);
  const height = Math.round(cropHeight * scale);
  return {
    width: Math.min(width, Math.floor(mediaSize.width)),
    height: Math.min(height, Math.floor(mediaSize.height)),
  };
}

function previewCanvasSize(cropWidth: number, cropHeight: number): { width: number; height: number } {
  const scale = PREVIEW_MAX / Math.max(cropWidth, cropHeight);
  return {
    width: Math.round(cropWidth * scale),
    height: Math.round(cropHeight * scale),
  };
}

interface CropperSession {
  mediaSize: MediaSize;
  cropSize: { width: number; height: number };
  crop: { x: number; y: number };
  zoom: number;
  initialArea: Area;
  cropWidth: number;
  cropHeight: number;
}

function drawCropPreview(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  area: Area,
  outputWidth: number,
  outputHeight: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  ctx.clearRect(0, 0, outputWidth, outputHeight);
  ctx.drawImage(image, area.x, area.y, area.width, area.height, 0, 0, outputWidth, outputHeight);
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

  const cropSize = meta ? cropDimensions(meta, targetResolution) : { width: targetResolution, height: targetResolution };
  const previewSize = previewCanvasSize(cropSize.width, cropSize.height);

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
        const dims = cropDimensions(loaded, targetResolution);
        setCroppedAreaPixels(centerToArea(loaded, dims.width, dims.height));
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

  const updatePreview = useCallback(
    (area: Area | null) => {
      const canvas = previewCanvasRef.current;
      const image = sourceImageRef.current;
      if (!canvas || !image || !area) return;
      drawCropPreview(canvas, image, area, previewSize.width, previewSize.height);
    },
    [previewSize.width, previewSize.height]
  );

  useEffect(() => {
    updatePreview(croppedAreaPixels);
  }, [croppedAreaPixels, sourceImageReady, updatePreview]);

  const handleMediaLoaded = useCallback(
    (mediaSize: MediaSize) => {
      if (!meta || session) return;
      const dims = cropDimensions(meta, targetResolution);
      const displayCropSize = computeDisplayCropSize(mediaSize, dims.width, dims.height);
      const initialArea = centerToArea(meta, dims.width, dims.height);
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
        cropWidth: dims.width,
        cropHeight: dims.height,
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
    if (!meta || !croppedAreaPixels || !session) return;
    setSaving(true);
    setError(null);
    try {
      const normalizedArea: Area = {
        x: croppedAreaPixels.x,
        y: croppedAreaPixels.y,
        width: session.cropWidth,
        height: session.cropHeight,
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

  const aspect = cropSize.width / cropSize.height;

  return (
    <Modal open onClose={onClose} size="xl" className="p-0 overflow-hidden max-w-4xl">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div>
          <div className="text-sm font-medium text-text">Crop image</div>
          <div className="text-xs text-text-muted truncate max-w-md">
            {filename}
            {meta?.enable_bucket && meta.bucket_width && meta.bucket_height
              ? ` · ${meta.bucket_width}×${meta.bucket_height}`
              : ` · ${targetResolution}×${targetResolution}`}
          </div>
        </div>
        <Button variant="icon" onClick={onClose} aria-label="Close">
          <X size={18} />
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-0">
        <div className="relative h-[min(60vh,520px)] bg-neutral-900">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-text-muted text-sm">
              Loading…
            </div>
          ) : meta && session ? (
            <Cropper
              image={imageUrl}
              crop={crop}
              zoom={zoom}
              aspect={aspect}
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
              aspect={aspect}
              onCropChange={() => {}}
              onMediaLoaded={handleMediaLoaded}
              restrictPosition
              objectFit="contain"
              style={{ containerStyle: { visibility: "hidden" } }}
            />
          ) : null}
          {!loading && meta && !session && (
            <div className="absolute inset-0 flex items-center justify-center text-text-muted text-sm">
              Preparing crop…
            </div>
          )}
        </div>

        <div className="border-t md:border-t-0 md:border-l border-border p-4 flex md:flex-col items-center justify-center gap-2 bg-bg">
          <div className="text-xs text-text-muted">Result preview</div>
          <canvas
            ref={previewCanvasRef}
            className="rounded-lg border border-border bg-black"
            style={{ width: previewSize.width, height: previewSize.height }}
          />
          <div className="text-[10px] text-text-muted text-center max-w-[160px]">
            Updates as you move the image under the crop frame
          </div>
        </div>
      </div>

      <div className="px-4 py-3 space-y-3 border-t border-border">
        <div className="flex items-center gap-3">
          <label className="text-xs text-text-muted shrink-0">Zoom</label>
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
        {error && <div className="text-sm text-error">{error}</div>}
        <ModalFooter className="justify-end pt-0">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="button" onClick={handleSave} disabled={saving || !session}>
            {saving ? "Saving…" : "Save & bake"}
          </Button>
        </ModalFooter>
      </div>
    </Modal>
  );
}
