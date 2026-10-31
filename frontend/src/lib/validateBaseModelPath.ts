import { filesApi } from "@/lib/api/files";
import { getBaseModelPathFormatIssue } from "@/lib/validateBaseModelPathFormat";

export { getBaseModelPathFormatIssue } from "@/lib/validateBaseModelPathFormat";

/** Validates format and that the file exists (via backend). */
export async function validateBaseModelPath(path: string): Promise<string | null> {
  const formatIssue = getBaseModelPathFormatIssue(path);
  if (formatIssue) return formatIssue;
  const result = await filesApi.validateBaseModel(path.trim());
  return result.valid ? null : result.message ?? "Base model file not found.";
}
