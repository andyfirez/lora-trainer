import { BASE_URL } from "@/lib/api/client";
import type { PngInfoResponse } from "@/types";

export const pngInfoApi = {
  async inspect(file: File): Promise<PngInfoResponse> {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${BASE_URL}/png-info`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
    }

    return res.json();
  },
};
