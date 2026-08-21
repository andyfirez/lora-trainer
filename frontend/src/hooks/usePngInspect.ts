"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { pngInfoApi } from "@/lib/api/pngInfo";
import type { PngInfoResponse } from "@/types";

export interface PngInspectState {
  dragActive: boolean;
  loading: boolean;
  error: string | null;
  fileName: string | null;
  localPreviewUrl: string | null;
  result: PngInfoResponse | null;
  copied: boolean;
}

const initialState: PngInspectState = {
  dragActive: false,
  loading: false,
  error: null,
  fileName: null,
  localPreviewUrl: null,
  result: null,
  copied: false,
};

type PngInspectAction =
  | { type: "SET_DRAG_ACTIVE"; active: boolean }
  | { type: "START_INSPECT"; fileName: string; previewUrl: string }
  | { type: "SUCCESS"; result: PngInfoResponse }
  | { type: "ERROR"; error: string }
  | { type: "SET_COPIED"; copied: boolean };

function pngInspectReducer(state: PngInspectState, action: PngInspectAction): PngInspectState {
  switch (action.type) {
    case "SET_DRAG_ACTIVE":
      return { ...state, dragActive: action.active };
    case "START_INSPECT":
      return {
        ...state,
        loading: true,
        error: null,
        copied: false,
        fileName: action.fileName,
        result: null,
        localPreviewUrl: action.previewUrl,
      };
    case "SUCCESS":
      return { ...state, loading: false, result: action.result };
    case "ERROR":
      return { ...state, loading: false, error: action.error };
    case "SET_COPIED":
      return { ...state, copied: action.copied };
    default:
      return state;
  }
}

export function usePngInspect() {
  const [state, dispatch] = useReducer(pngInspectReducer, initialState);
  const previewUrlRef = useRef<string | null>(null);

  useEffect(() => {
    previewUrlRef.current = state.localPreviewUrl;
  }, [state.localPreviewUrl]);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const inspectFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      dispatch({ type: "ERROR", error: "Please choose an image file." });
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    dispatch({ type: "START_INSPECT", fileName: file.name, previewUrl: objectUrl });

    try {
      const response = await pngInfoApi.inspect(file);
      dispatch({ type: "SUCCESS", result: response });
    } catch (err: unknown) {
      dispatch({
        type: "ERROR",
        error: err instanceof Error ? err.message : "Failed to inspect image",
      });
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) void inspectFile(file);
    },
    [inspectFile],
  );

  const setDragActive = useCallback((active: boolean) => {
    dispatch({ type: "SET_DRAG_ACTIVE", active });
  }, []);

  const copyRawInfo = useCallback(async () => {
    if (!state.result?.info) return;
    try {
      await navigator.clipboard.writeText(state.result.info);
      dispatch({ type: "SET_COPIED", copied: true });
      window.setTimeout(() => dispatch({ type: "SET_COPIED", copied: false }), 2000);
    } catch {
      dispatch({ type: "ERROR", error: "Failed to copy to clipboard" });
    }
  }, [state.result?.info]);

  return {
    ...state,
    inspectFile,
    handleFiles,
    setDragActive,
    copyRawInfo,
  };
}
