import type { GpuDefaultsInfo } from "@/lib/api/settings";

const GPU_GLOBAL_KEYS = ["tf32", "attention_mechanism"] as const;
const GPU_OVERRIDE_KEYS = ["mixed_precision", "vae_dtype", "sample_vae_tiling"] as const;

export function applySparseGpuOverrides(
  config: Record<string, unknown>,
  gpuDefaults: GpuDefaultsInfo,
): Record<string, unknown> {
  const next = { ...config };
  for (const key of GPU_GLOBAL_KEYS) {
    delete next[key];
  }
  for (const key of GPU_OVERRIDE_KEYS) {
    if (!(key in next)) continue;
    const value = next[key];
    if (value == null) {
      delete next[key];
      continue;
    }
    if (key === "mixed_precision" && value === gpuDefaults.mixed_precision) {
      delete next[key];
    } else if (key === "vae_dtype" && value === gpuDefaults.vae_dtype) {
      delete next[key];
    } else if (key === "sample_vae_tiling" && value === gpuDefaults.sample_vae_tiling) {
      delete next[key];
    }
  }
  return next;
}

export const MIXED_PRECISION_OPTIONS = [
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

export const VAE_DTYPE_OPTIONS = [
  { value: "auto", label: "auto" },
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];
