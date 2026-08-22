export interface GalleryEntry {
  key: string;
  url: string;
  kind: "preview" | "cell" | "grid";
  samplingId: number;
  infotext: string;
  seed: number | null;
  params?: Record<string, unknown>;
}
