import { BASE_URL } from "@/lib/api/client";
import type { PickPathRequest } from "@/types";

export type { PickKind, PickPathRequest } from "@/types";

export const filesApi = {
  pick: async (body: PickPathRequest): Promise<string | null> => {
    const res = await fetch(`${BASE_URL}/files/pick`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 204) return null;
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${res.status}`);
    }
    const data = (await res.json()) as { path: string };
    return data.path;
  },
};
