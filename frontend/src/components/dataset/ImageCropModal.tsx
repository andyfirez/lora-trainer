"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Cropper, { type Area, type MediaSize, getInitialCropFromCroppedAreaPixels } from "react-easy-crop";
import Modal, { ModalError, ModalFooter } from "@/components/ui/Modal";
import Button from "@/components/ui/Button";
import { labelClassName } from "@/components/ui/Input";
import { datasetCropPreviewUrl, datasetsApi } from "@/lib/api/datasets";
import {
  areaToCenter,
  centerToArea,
  computeDisplayCropSize,
  cropDimensions,
  normalizeSelectionForSave,
  previewCanvasSize,
  selectionToInitialArea,
} from "@/lib/imageCropUtils";
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
  outputWidth: number,
  outputHeight: number,
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
  const croppedAreaPixelsRef = useRef<Area | null>(null);
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

  const initialCroppedAreaPixels = useMemo(() => {
    if (!meta) return undefined;
    return selectionToInitialArea(meta) ?? centerToArea(meta, cropSize.width, cropSize.height);
  }, [meta, cropSize.width, cropSize.height]);

  const cropDescription = meta
    ? `${filename}${
        meta.enable_bucket && meta.bucket_width && meta.bucket_height
          ? ` · ${meta.bucket_width}×${meta.bucket_height}`
          : ` · ${targetResolution}×${targetResolution}`
      }`
    : filename;

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
        const initialArea = selectionToInitialArea(loaded) ?? centerToArea(loaded, dims.width, dims.height);
        croppedAreaPixelsRef.current = initialArea;
        setCroppedAreaPixels(initialArea);
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
    [previewSize.width, previewSize.height],
  );

  useEffect(() => {
    updatePreview(croppedAreaPixels);
  }, [croppedAreaPixels, sourceImageReady, updatePreview]);

  const handleMediaLoaded = useCallback(
    (mediaSize: MediaSize) => {
      if (!meta || session || !initialCroppedAreaPixels) return;
      const dims = cropDimensions(meta, targetResolution);
      const displayCropSize = computeDisplayCropSize(mediaSize, dims.width, dims.height);
      const initial = getInitialCropFromCroppedAreaPixels(
        initialCroppedAreaPixels,
        mediaSize,
        0,
        displayCropSize,
        MIN_ZOOM,
        MAX_ZOOM,
      );
      setSession({
        mediaSize,
        cropSize: displayCropSize,
        crop: initial.crop,
        zoom: initial.zoom,
        initialArea: initialCroppedAreaPixels,
      });
      setCrop(initial.crop);
      setZoom(initial.zoom);
      croppedAreaPixelsRef.current = initialCroppedAreaPixels;
      setCroppedAreaPixels(initialCroppedAreaPixels);
    },
    [initialCroppedAreaPixels, meta, session, targetResolution],
  );

  const handleCropAreaChange = useCallback((_cropped: Area, pixels: Area) => {
    croppedAreaPixelsRef.current = pixels;
    setCroppedAreaPixels(pixels);
  }, []);

  const handleSave = async () => {
    const pixels = croppedAreaPixelsRef.current;
    if (!meta || !pixels || !session) return;
    setSaving(true);
    setError(null);
    try {
      const selection = normalizeSelectionForSave(pixels);
      const center = areaToCenter(pixels, meta);
      await datasetsApi.saveCrop(datasetId, filename, {
        crop_center_x: center.x,
        crop_center_y: center.y,
        ...selection,
      });
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
    <Modal
      open
      onClose={onClose}
      title="Crop image"
      description={cropDescription}
      className="max-w-4xl p-0 gap-0 overflow-hidden"
    >
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-0 -mx-6">
        <div className="relative h-[min(60vh,520px)] bg-neutral-900">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">
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
              roundCropAreaPixels
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
            <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">
              Preparing crop…
            </div>
          )}
        </div>

        <div className="border-t md:border-t-0 md:border-l border-border p-4 flex md:flex-col items-center justify-center gap-2 bg-bg">
          <div className="text-xs text-muted">Result preview</div>
          <canvas
            ref={previewCanvasRef}
            className="rounded-lg border border-border bg-black"
            style={{ width: previewSize.width, height: previewSize.height }}
          />
          <div className="text-[10px] text-muted text-center max-w-[160px]">
            Updates as you move the image under the crop frame
          </div>
        </div>
      </div>

      <div className="px-0 pt-4 space-y-3">
        <div className="flex items-center gap-3">
          <label className={`${labelClassName} mb-0 shrink-0`}>Zoom</label>
          <input
            type="range"
            min={MIN_ZOOM}
            max={MAX_ZOOM}
            step={0.05}
            value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))}
            className="flex-1 accent-accent"
            disabled={!session}
          />
        </div>
        {error && <ModalError>{error}</ModalError>}
        <ModalFooter>
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
