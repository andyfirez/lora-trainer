import { BASE_URL } from "@/lib/api/client";

/** Resolve API-relative media path (e.g. /loras/1/samples/foo.png) to a fetchable URL. */
export function mediaUrl(path: string): string {
  return `${BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}
