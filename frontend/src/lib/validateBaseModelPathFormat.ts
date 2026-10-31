const CHECKPOINT_EXT = /\.(safetensors|ckpt)$/i;

/** Client-side format checks before hitting the API. */
export function getBaseModelPathFormatIssue(path: string): string | null {
  const trimmed = path.trim();
  if (!trimmed) {
    return "Select an SDXL checkpoint file (.safetensors or .ckpt).";
  }
  if (!CHECKPOINT_EXT.test(trimmed)) {
    return "Base model must be a local .safetensors or .ckpt file (HuggingFace IDs and folders are not supported).";
  }
  return null;
}
