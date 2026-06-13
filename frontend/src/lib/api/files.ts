const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export type PickKind = "file" | "directory" | "model";

export interface PickPathRequest {
  kind?: PickKind;
  title?: string;
  initial_path?: string;
}

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
