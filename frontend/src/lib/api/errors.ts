export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function formatValidationDetail(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((item) => {
      if (item && typeof item === "object" && "msg" in item) {
        return String((item as { msg: unknown }).msg);
      }
      return null;
    })
    .filter((msg): msg is string => Boolean(msg));
  return messages.length ? messages.join(", ") : null;
}

export function parseApiError(body: unknown, status: number): ApiError {
  const record = body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  const detail = record.detail;
  const validationMessage = formatValidationDetail(detail);
  const message =
    validationMessage ??
    (typeof detail === "string" ? detail : null) ??
    (typeof record.message === "string" ? record.message : null) ??
    `HTTP ${status}`;
  return new ApiError(message, status, detail);
}

const UNREACHABLE_API_MESSAGE = "Cannot reach the API. Is the backend running?";

export function parseNetworkError(err: unknown): ApiError {
  const message = err instanceof Error ? err.message : "Network error";
  if (err instanceof TypeError && (message === "Failed to fetch" || message === "NetworkError when attempting to fetch resource.")) {
    return new ApiError(UNREACHABLE_API_MESSAGE, 0, err);
  }
  return new ApiError(message, 0, err);
}
